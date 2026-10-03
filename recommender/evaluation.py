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
"""
from dataclasses import dataclass

import numpy as np

from .collaborative import EASE
from .content import top_k
from .hybrid import HybridRecommender

L2_GRID = (10, 50, 100, 200, 500, 1000, 2000)
ALPHA_GRID = tuple(round(float(a), 2) for a in np.linspace(0, 1, 11))
BETA_GRID = (0.0, 0.05, 0.1, 0.15)


@dataclass
class EvalUser:
    fold_in: np.ndarray
    held_out: np.ndarray


@dataclass
class Split:
    train_rows: np.ndarray
    val: list
    test: list
    val_rows: np.ndarray
    test_rows: np.ndarray


def split_users(X, min_likes=5, val_frac=0.15, test_frac=0.15, holdout=0.2, seed=0):
    rng = np.random.default_rng(seed)
    likes_per_user = np.diff(X.indptr)
    eligible = np.flatnonzero(likes_per_user >= min_likes)
    rng.shuffle(eligible)
    n_val, n_test = int(len(eligible) * val_frac), int(len(eligible) * test_frac)
    val_rows, test_rows = eligible[:n_val], eligible[n_val:n_val + n_test]
    eval_rows = set(val_rows) | set(test_rows)
    train_rows = np.array([u for u in range(X.shape[0]) if u not in eval_rows])

    def _hide(rows):
        users = []
        for u in rows:
            items = X.indices[X.indptr[u]:X.indptr[u + 1]].copy()
            rng.shuffle(items)
            n_hide = max(1, int(round(len(items) * holdout)))
            users.append(EvalUser(fold_in=np.sort(items[n_hide:]), held_out=items[:n_hide]))
        return users

    return Split(train_rows, _hide(val_rows), _hide(test_rows), val_rows, test_rows)


def evaluate(score_fn, users, n_items, k=10):
    """score_fn(fold_in ids) -> score array over the catalog."""
    recalls, ndcgs, shown = [], [], set()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for user in users:
        recs = top_k(score_fn(list(user.fold_in)), user.fold_in, k)
        shown.update(recs.tolist())
        hits = np.isin(recs, user.held_out)
        recalls.append(hits.sum() / min(k, len(user.held_out)))
        ideal = discounts[:min(k, len(user.held_out))].sum()
        ndcgs.append((hits * discounts[:len(hits)]).sum() / ideal)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "coverage": len(shown) / n_items,
    }


def model_scorers(content, collab, alpha, beta):
    """Score functions for every model we compare."""
    hybrid = HybridRecommender(content, collab, alpha=alpha, beta=beta)
    pop = collab.item_counts.astype(float)
    return {
        "popularity": lambda liked: pop,
        "content": lambda liked: content.scores(liked),
        "collaborative (EASE)": lambda liked: collab.scores(liked),
        "hybrid": lambda liked: hybrid.scores(liked),
    }


def tune(content, X, split, k=10, l2_grid=L2_GRID, alpha_grid=ALPHA_GRID, beta_grid=BETA_GRID,
         log=print):
    """Grid-search EASE l2 and hybrid alpha/beta on validation users (by ndcg@k).

    Returns (params, val_metrics) with params = {"l2", "alpha", "beta"}.
    """
    best = None
    X_train = X[split.train_rows]
    for l2 in l2_grid:
        collab = EASE(l2).fit(X_train)
        for alpha in alpha_grid:
            for beta in beta_grid:
                hybrid = HybridRecommender(content, collab, alpha=alpha, beta=beta)
                m = evaluate(hybrid.scores, split.val, content.n_items, k)
                if best is None or m[f"ndcg@{k}"] > best[1][f"ndcg@{k}"]:
                    best = ({"l2": l2, "alpha": alpha, "beta": beta}, m)
        log(f"  l2={l2:<5} best so far: {best[0]} ndcg@{k}={best[1][f'ndcg@{k}']:.4f}")
    return best


def compare_on_test(content, X, split, l2, alpha, beta, k=10):
    """Fit on train+val users, report every model on the test users."""
    fit_rows = np.concatenate([split.train_rows, split.val_rows])
    collab = EASE(l2).fit(X[fit_rows])
    return {
        name: evaluate(fn, split.test, content.n_items, k)
        for name, fn in model_scorers(content, collab, alpha, beta).items()
    }


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
