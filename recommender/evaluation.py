"""Offline evaluation on held-out MovieLens users (strong generalization).

Protocol:
  1. Keep users with at least ``min_likes`` positive ratings in the catalog.
  2. Split users into train / validation / test groups (disjoint).
  3. For each val/test user, hide ``holdout`` of their likes; the rest are the
     "fold-in" likes the model gets as input (like picking favorites in the app).
  4. Each model ranks the catalog (minus fold-in likes); we check how many
     hidden likes land in the top K.

Metrics (averaged over users):
  recall@K   hits / min(K, #hidden)
  ndcg@K     rank-discounted hits, normalized by the ideal ranking
  coverage   share of the catalog that appears in anyone's top K

All scoring is batched: a score function receives a sparse (users x items)
fold-in batch plus that batch's slice of the eval set, and returns dense
(users x items) scores, so tens of thousands of users evaluate in seconds.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags

from .collaborative import EASE
from .hybrid import blend

L2_GRID = (50, 100, 200, 500, 1000, 2000, 5000)
ALPHA_GRID = tuple(round(float(a), 2) for a in np.linspace(0, 1, 11))
BETA_GRID = (0.0, 0.05, 0.1, 0.15)
BATCH = 2048


@dataclass
class EvalSet:
    rows: np.ndarray        # user rows in X
    fold_in: csr_matrix     # likes given to the model
    held_out: csr_matrix    # hidden likes to recover

    def __len__(self):
        return len(self.rows)


@dataclass
class Split:
    train_rows: np.ndarray
    val: EvalSet
    test: EvalSet

    @property
    def fit_rows(self):
        """Users a final model may train on when reporting test metrics."""
        return np.sort(np.concatenate([self.train_rows, self.val.rows]))


def _hide(X, rows, holdout, rng):
    sub = X[rows].tocoo()
    order = np.lexsort((rng.random(sub.nnz), sub.row))  # shuffle likes within each user
    r, c = sub.row[order], sub.col[order]
    counts = np.bincount(r, minlength=len(rows))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_hide = np.maximum(1, np.round(counts * holdout).astype(int))
    hidden = (np.arange(len(r)) - starts[r]) < n_hide[r]

    def _mat(mask):
        return csr_matrix((np.ones(mask.sum(), dtype=np.float32), (r[mask], c[mask])), shape=(len(rows), X.shape[1]))

    return EvalSet(rows, _mat(~hidden), _mat(hidden))


def split_users(X, min_likes=5, val_frac=0.15, test_frac=0.15, holdout=0.2, seed=0,
                max_val=5000, max_test=20000):
    """Disjoint train/val/test users; val/test capped to keep evaluation fast."""
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(np.diff(X.indptr) >= min_likes)
    rng.shuffle(eligible)
    n_val = min(int(len(eligible) * val_frac), max_val)
    n_test = min(int(len(eligible) * test_frac), max_test)
    val_rows, test_rows = np.sort(eligible[:n_val]), np.sort(eligible[n_val:n_val + n_test])
    train_mask = np.ones(X.shape[0], dtype=bool)
    train_mask[val_rows] = train_mask[test_rows] = False
    return Split(np.flatnonzero(train_mask), _hide(X, val_rows, holdout, rng), _hide(X, test_rows, holdout, rng))


def batches(n, size=BATCH):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def evaluate(score_fn, users: EvalSet, k=10):
    """score_fn(F_batch, batch_slice) -> (batch x items) or (items,) scores."""
    n_items = users.fold_in.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)
    recalls, ndcgs, shown = [], [], np.zeros(n_items, dtype=bool)
    for b in batches(len(users)):
        F, H = users.fold_in[b], users.held_out[b]
        scores = np.array(np.broadcast_to(score_fn(F, b), (F.shape[0], n_items)), dtype=np.float32)
        rows, cols = F.nonzero()
        scores[rows, cols] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1), 1)
        shown[top.ravel()] = True
        hits = np.asarray(H[np.arange(H.shape[0])[:, None], top].todense()) > 0
        n_rel = np.minimum(k, np.asarray(H.sum(axis=1)).ravel()).astype(int)
        recalls.append(hits.sum(1) / n_rel)
        ndcgs.append((hits * discounts).sum(1) / ideal[n_rel - 1])
    return {
        f"recall@{k}": float(np.concatenate(recalls).mean()),
        f"ndcg@{k}": float(np.concatenate(ndcgs).mean()),
        "coverage": float(shown.mean()),
    }


def precompute(fn, users: EvalSet):
    """Run a batch score function over an eval set once; returns (users x items)."""
    return np.vstack([fn(users.fold_in[b]) for b in batches(len(users))])


def tune(content, X, split, k=10, l2_grid=L2_GRID, alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID, log=print):
    """Grid-search EASE l2 and hybrid alpha/beta on validation users (by ndcg@k).

    Content similarity and raw EASE scores are computed once per l2, so the
    alpha/beta sweep is just array math. Returns (params, val_metrics).
    """
    best = None
    X_train = X[split.train_rows]
    sim = precompute(content.similarity_batch, split.val)
    for l2 in l2_grid:
        collab = EASE(l2).fit(X_train)
        cf = precompute(collab.scores_batch, split.val)
        for alpha in alpha_grid:
            for beta in beta_grid:
                def score(F, b, alpha=alpha, beta=beta, cf=cf, covered=collab.scored):
                    return blend(sim[b], cf[b], F, covered, content.quality, alpha, beta)
                m = evaluate(score, split.val, k)
                if best is None or m[f"ndcg@{k}"] > best[1][f"ndcg@{k}"]:
                    best = ({"l2": l2, "alpha": alpha, "beta": beta}, m)
        log(f"  l2={l2:<5} best so far: {best[0]} ndcg@{k}={best[1][f'ndcg@{k}']:.4f}")
    return best


def baseline_scorers(content, collab, alpha, beta):
    """Batch score functions (F, slice) -> scores for the models we compare."""
    pop = collab.item_counts.astype(np.float32)
    return {
        "popularity": lambda F, b: pop,
        "content": lambda F, b: content.blend_quality(content.similarity_batch(F)),
        "collaborative (EASE)": lambda F, b: collab.scores_batch(F),
        "hybrid": lambda F, b: blend(content.similarity_batch(F), collab.scores_batch(F), F,
                                     collab.scored, content.quality, alpha, beta),
    }


def compare_on_test(content, X, split, l2, alpha, beta, k=10):
    """Fit on train+val users, report every model on the test users."""
    collab = EASE(l2).fit(X[split.fit_rows])
    return {name: evaluate(fn, split.test, k) for name, fn in baseline_scorers(content, collab, alpha, beta).items()}


def aggregate(runs):
    """Mean and std per model/metric across several compare_on_test results."""
    mean, std = {}, {}
    for name in runs[0]:
        mean[name] = {m: float(np.mean([r[name][m] for r in runs])) for m in runs[0][name]}
        std[name] = {m: float(np.std([r[name][m] for r in runs])) for m in runs[0][name]}
    return mean, std


def format_table(results, std=None):
    metrics = list(next(iter(results.values())))
    width = max(len(n) for n in results) + 2
    col = 18 if std else 12
    lines = [f"{'model':<{width}}" + "".join(f"{m:>{col}}" for m in metrics)]
    lines.append("-" * len(lines[0]))
    for name, m in results.items():
        if std:
            cells = "".join(f"{f'{m[x]:.4f} ± {std[name][x]:.4f}':>{col}}" for x in metrics)
        else:
            cells = "".join(f"{m[x]:>{col}.4f}" for x in metrics)
        lines.append(f"{name:<{width}}" + cells)
    return "\n".join(lines)


# ---- cold start --------------------------------------------------------------

def pick_cold_items(X_train, frac=0.1, min_likes=50, seed=0):
    """Random sample of well-liked catalog items to treat as "new releases"."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(X_train.sum(axis=0)).ravel()
    pool = np.flatnonzero(counts >= min_likes)
    return np.sort(rng.choice(pool, size=min(int(X_train.shape[1] * frac), len(pool)), replace=False))


def without_items(X, items):
    """Copy of X with the given item columns emptied (no collaborative signal)."""
    keep = np.ones(X.shape[1], dtype=np.float32)
    keep[items] = 0
    X = (X @ diags(keep)).tocsr()
    X.eliminate_zeros()
    return X


def cold_start_users(users: EvalSet, cold_items):
    """Eval users who have at least one hidden like among the cold items, with
    held-out restricted to those items."""
    mask = np.zeros(users.held_out.shape[1], dtype=np.float32)
    mask[cold_items] = 1
    held = (users.held_out @ diags(mask)).tocsr()
    held.eliminate_zeros()
    keep = np.flatnonzero(np.diff(held.indptr) > 0)
    return EvalSet(users.rows[keep], users.fold_in[keep], held[keep])


def evaluate_cold(score_fn, users: EvalSet, cold_items, k=10):
    """Rank only the cold items: how well does a model surface new releases a
    user will like, without any ratings for them?"""
    not_cold = np.ones(users.fold_in.shape[1], dtype=bool)
    not_cold[cold_items] = False

    def restricted(F, b):
        scores = np.array(np.broadcast_to(score_fn(F, b), F.shape), dtype=np.float32)
        scores[:, not_cold] = -np.inf
        return scores

    m = evaluate(restricted, users, k)
    m.pop("coverage")
    return m
