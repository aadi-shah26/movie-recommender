"""Two-stage ranking: candidate generation + LightGBM LambdaRank reranker.

Stage 1 (retrieval): for each user, take the top ``n_collab`` movies by EASE
score plus the top ``n_content`` by content similarity that EASE didn't
already pick (so cold-start movies with no ratings can still surface).

Stage 2 (ranking): a gradient-boosted LambdaRank model scores every candidate
from features describing the user, the movie and how they relate: EASE score
and rank, content similarity (overall, per feature block, and to the single
closest liked movie), popularity, rating, year, and user-level stats. It is
trained on held-out likes of validation users, optimizing NDCG directly.
"""
import lightgbm as lgb
import numpy as np
from scipy.sparse import diags
from sklearn.preprocessing import normalize

from .evaluation import EvalSet, batches

FEATURES = [
    "ease_raw", "ease_rescaled", "ease_rank",
    "ease_max_link", "cooc_max", "cooc_mean",
    "sim_profile", "sim_rank", "sim_max", "sim_text", "sim_genre", "sim_director", "sim_actors",
    "item_log_pop", "item_covered", "item_rating", "item_log_votes", "item_metascore", "item_year",
    "item_ml_rating", "item_like_rate",
    "user_n_likes", "user_mean_log_pop", "user_mean_year",
    "pop_diff", "year_diff",
]
BLOCK_FEATURES = ("text", "genre", "director", "actors")

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "lambdarank_truncation_level": 20,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 0,
}


def item_stats_from_ratings(ratings, user_ids):
    """Per-item MovieLens mean rating and like rate, from ``user_ids`` only.

    ``ratings`` is a CatalogRatings; restrict ``user_ids`` to training users so
    a test user's own hidden ratings never leak into the features.
    """
    keep = np.isin(ratings.users, user_ids)
    items, values = ratings.items[keep], ratings.ratings[keep]
    n = ratings.n_items
    count = np.bincount(items, minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "ml_rating": np.bincount(items, values, minlength=n) / count,
            "like_rate": np.bincount(items, values >= 4.0, minlength=n) / count,
        }


def _top_cols(scores, k):
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1), 1)


class Reranker:
    def __init__(self, content, collab, years=None, item_stats=None, n_collab=100, n_content=20, booster=None):
        """item_stats: optional {"ml_rating", "like_rate"} arrays per catalog item,
        computed from training users only (see item_stats_from_ratings)."""
        self.content, self.collab = content, collab
        self.n_collab, self.n_content = n_collab, n_content
        self.booster = booster
        df = content.df
        n = content.n_items
        years = np.asarray(years if years is not None else df["year"].fillna(-1), dtype=np.float32)
        self.item = {
            "log_pop": np.log1p(collab.item_counts).astype(np.float32),
            "covered": collab.covered.astype(np.float32),
            "rating": df["rating"].to_numpy(np.float32),
            "log_votes": np.log1p(df["votes"].to_numpy(np.float32)),
            "metascore": df["metascore"].to_numpy(np.float32),
            "year": np.where(years > 0, years, np.nan).astype(np.float32),
            "ml_rating": np.full(n, np.nan, dtype=np.float32),
            "like_rate": np.full(n, np.nan, dtype=np.float32),
        }
        for key, values in (item_stats or {}).items():
            self.item[key] = np.asarray(values, dtype=np.float32)
        # Item-item cosine similarity, for "closest liked movie" features.
        self.item_sim = content.item_similarity()
        np.fill_diagonal(self.item_sim, 0.0)
        assert all(len(v) == n for v in self.item.values())

    # ---- features ------------------------------------------------------------

    def _block_sims(self, F):
        counts = np.asarray(F.sum(axis=1)).ravel()
        mean = diags(1.0 / np.maximum(counts, 1)) @ F
        out = {}
        for name in BLOCK_FEATURES:
            block = self.content.blocks.get(name)
            if block is None:
                out[name] = np.zeros(F.shape, dtype=np.float32)
                continue
            profile = normalize(mean @ block, norm="l2", axis=1)
            out[name] = (profile @ block.T).toarray().astype(np.float32)
        return out

    @staticmethod
    def _max_over_likes(F, item_item):
        """Row u: for every item, the max of item_item[liked, item] over u's likes."""
        out = np.zeros(F.shape, dtype=np.float32)
        for u in range(F.shape[0]):
            liked = F.indices[F.indptr[u]:F.indptr[u + 1]]
            if len(liked):
                out[u] = item_item[liked].max(axis=0)
        return out

    def candidates_and_features(self, F):
        """Returns (candidate ids (users x C), features (users*C x len(FEATURES)))."""
        n_users, n_items = F.shape
        rows, cols = F.nonzero()
        cf = self.collab.scores_batch(F)
        sim = self.content.similarity_batch(F)

        cf_masked, sim_masked = cf.copy(), sim.copy()
        cf_masked[rows, cols] = -np.inf
        sim_masked[rows, cols] = -np.inf
        cand_cf = _top_cols(cf_masked, self.n_collab)
        sim_masked[np.arange(n_users)[:, None], cand_cf] = -np.inf
        cand = np.hstack([cand_cf, _top_cols(sim_masked, self.n_content)])

        # Ranks within the user's full catalog ordering (0 = best).
        cf_rank = np.argsort(np.argsort(-cf_masked, axis=1), axis=1).astype(np.float32)
        sim_full = sim.copy()
        sim_full[rows, cols] = -np.inf
        sim_rank = np.argsort(np.argsort(-sim_full, axis=1), axis=1).astype(np.float32)

        peak = np.where(self.collab.scored, cf_masked, -np.inf).max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cf_rescaled = np.where(peak > 0, cf / peak, np.nan)

        blocks = self._block_sims(F)
        max_sim = self._max_over_likes(F, self.item_sim)
        max_link = self._max_over_likes(F, self.collab.B)
        cooc_max = self._max_over_likes(F, self.collab.cooc)

        counts = np.asarray(F.sum(axis=1)).ravel()
        cooc_mean = np.asarray(F @ self.collab.cooc) / np.maximum(counts, 1)[:, None]
        user_pop = (F @ self.item["log_pop"]) / np.maximum(counts, 1)
        year_filled = np.nan_to_num(self.item["year"])
        has_year = (~np.isnan(self.item["year"])).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            user_year = (F @ year_filled) / (F @ has_year)

        def gather(mat):
            return np.take_along_axis(mat, cand, 1)

        def item(name):
            return self.item[name][cand]

        def user(vec):
            return np.repeat(np.asarray(vec, dtype=np.float32)[:, None], cand.shape[1], axis=1)

        cols_ = {
            "ease_raw": gather(cf), "ease_rescaled": gather(cf_rescaled), "ease_rank": gather(cf_rank),
            "ease_max_link": gather(max_link), "cooc_max": gather(cooc_max), "cooc_mean": gather(cooc_mean),
            "sim_profile": gather(sim), "sim_rank": gather(sim_rank), "sim_max": gather(max_sim),
            **{f"sim_{b}": gather(blocks[b]) for b in BLOCK_FEATURES},
            "item_log_pop": item("log_pop"), "item_covered": item("covered"), "item_rating": item("rating"),
            "item_log_votes": item("log_votes"), "item_metascore": item("metascore"), "item_year": item("year"),
            "item_ml_rating": item("ml_rating"), "item_like_rate": item("like_rate"),
            "user_n_likes": user(counts), "user_mean_log_pop": user(user_pop), "user_mean_year": user(user_year),
        }
        cols_["pop_diff"] = cols_["item_log_pop"] - cols_["user_mean_log_pop"]
        cols_["year_diff"] = cols_["item_year"] - cols_["user_mean_year"]
        X = np.stack([cols_[f].astype(np.float32).ravel() for f in FEATURES], axis=1)
        return cand, X

    def _dataset(self, users: EvalSet):
        Xs, ys, groups = [], [], []
        for b in batches(len(users), 1024):
            cand, X = self.candidates_and_features(users.fold_in[b])
            H = users.held_out[b]
            y = (np.asarray(H[np.arange(H.shape[0])[:, None], cand].todense()) > 0)
            keep = y.any(axis=1)  # users whose hidden likes were all missed teach nothing
            C = cand.shape[1]
            Xs.append(X.reshape(len(cand), C, -1)[keep].reshape(-1, X.shape[1]))
            ys.append(y[keep].ravel().astype(np.float32))
            groups += [C] * int(keep.sum())
        return np.vstack(Xs), np.concatenate(ys), groups

    def candidate_recall(self, users: EvalSet):
        """Share of hidden likes that make it into the candidate set (stage-1 ceiling)."""
        hit = total = 0
        for b in batches(len(users), 1024):
            cand, _ = self.candidates_and_features(users.fold_in[b])
            H = users.held_out[b]
            hit += int(np.asarray(H[np.arange(H.shape[0])[:, None], cand].todense()).sum())
            total += H.nnz
        return hit / total

    # ---- training / scoring --------------------------------------------------

    def fit(self, train: EvalSet, valid: EvalSet, extra_train=(), params=None, rounds=1000, log=print):
        """Train the booster. ``extra_train`` is a list of (Reranker, EvalSet)
        whose candidates/features come from a different stage-1 model, e.g. one
        with some movies hidden so the ranker learns to handle cold start."""
        parts = [self._dataset(train)] + [ranker._dataset(users) for ranker, users in extra_train]
        X_tr = np.vstack([p[0] for p in parts])
        y_tr = np.concatenate([p[1] for p in parts])
        g_tr = [g for p in parts for g in p[2]]
        X_va, y_va, g_va = self._dataset(valid)
        log(f"  ranker data: {len(g_tr):,} train / {len(g_va):,} valid queries, {len(X_tr):,} rows")
        dtrain = lgb.Dataset(X_tr, y_tr, group=g_tr, feature_name=FEATURES, free_raw_data=True)
        dvalid = lgb.Dataset(X_va, y_va, group=g_va, reference=dtrain)
        self.booster = lgb.train(
            {**DEFAULT_PARAMS, **(params or {})}, dtrain, rounds, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        log(f"  best iteration {self.booster.best_iteration}, valid ndcg@10 "
            f"{self.booster.best_score['valid_0']['ndcg@10']:.4f}")
        return self

    def scores_batch(self, F):
        """Dense (users x items) scores: reranked candidates first, the rest below."""
        cand, X = self.candidates_and_features(F)
        pred = self.booster.predict(X, num_iteration=self.booster.best_iteration or None)
        out = np.full(F.shape, -1e9, dtype=np.float32)
        np.put_along_axis(out, cand, pred.reshape(cand.shape).astype(np.float32), 1)
        return out

    def feature_importance(self):
        gain = self.booster.feature_importance("gain")
        total = gain.sum() or 1.0
        return dict(sorted(((f, float(g / total)) for f, g in zip(FEATURES, gain, strict=True)),
                           key=lambda kv: -kv[1]))
