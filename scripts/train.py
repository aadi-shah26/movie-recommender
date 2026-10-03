#!/usr/bin/env python3
"""Train the two-stage recommender and save it to models/.

Pipeline:
  1. Load MovieLens ratings matched to the catalog (cached after first run).
  2. Split users: train (EASE), ranker (tuning + LightGBM), test (report only).
  3. Tune EASE l2 + hybrid alpha/beta on a slice of the ranker users.
  4. Fit EASE on train users; impute scores for movies with no ratings.
  5. Train the LightGBM LambdaRank reranker on the ranker users.
  6. Report every model on held-out test users (all movies), and on a
     simulated cold-start benchmark (movies removed from collaborative data).
  7. Save the exact evaluated models + metrics to models/.

Run from the project root:
    python -m scripts.train                     # ML-32M (downloads ~240 MB once)
    python -m scripts.train --dataset ml-latest-small --quick
"""
import argparse
import time

import numpy as np

from recommender import movielens
from recommender.catalog import load_movie_df
from recommender.collaborative import EASE
from recommender.content import ContentRecommender
from recommender.evaluation import (
    EvalSet,
    baseline_scorers,
    cold_start_users,
    evaluate,
    evaluate_cold,
    format_table,
    pick_cold_items,
    split_users,
    tune,
    without_items,
)
from recommender.hybrid import HybridRecommender
from recommender.model import TrainedModel
from recommender.ranker import Reranker, item_stats_from_ratings

# Tuned on simulated cold-start items of validation users (see README):
# plot-embedding weight in the content model, and how cold movies borrow
# collaborative scores from their k most content-similar rated movies.
EMBEDDING_WEIGHT = 12.0
IMPUTE_K = 25
IMPUTE_POWER = 4.0
N_COLLAB, N_CONTENT = 100, 20
TUNE_USERS = 5000
RANKER_VALID_USERS = 5000
# Item dropout: this share of ranker training users see features from an EASE
# model with DROPOUT_FRAC of movies hidden, so the ranker learns cold start.
DROPOUT_USERS = 1 / 3
DROPOUT_FRAC = 0.1


def _slice(users: EvalSet, s: slice):
    return EvalSet(users.rows[s], users.fold_in[s], users.held_out[s])


def _hide_stats(stats, items):
    """Item stats as they'd look if ``items`` had no ratings."""
    return {k: np.where(np.isin(np.arange(len(v)), items), np.nan, v) for k, v in stats.items()}


def run(dataset=movielens.DEFAULT_DATASET, seed=0, quick=False, use_ranker=True, k=10, log=print):
    """Train + evaluate one split. Returns the model, test metrics and metadata."""
    df = load_movie_df()
    content = ContentRecommender(df, weights={"embedding": EMBEDDING_WEIGHT})
    item_sim = content.item_similarity()

    log(f"Loading MovieLens ({dataset})...")
    ratings = movielens.load_catalog_ratings(df["title"], dataset)
    X, user_ids = ratings.interactions()
    log(f"Matched {ratings.matched}/{len(df)} catalog movies; {X.shape[0]:,} users, {X.nnz:,} likes")

    max_val, max_test = (6000, 3000) if quick else (60000, 20000)
    split = split_users(X, val_frac=0.4, max_val=max_val, max_test=max_test, seed=seed)
    log(f"Users: {len(split.train_rows):,} train / {len(split.val):,} ranker+tuning / {len(split.test):,} test")
    X_train = X[split.train_rows]

    # -- tune EASE + hybrid blend ------------------------------------------------
    tune_split = type(split)(split.train_rows, _slice(split.val, slice(0, TUNE_USERS)), split.test)
    grids = {"l2_grid": (200, 500), "alpha_grid": (0.8, 0.9, 1.0), "beta_grid": (0.0, 0.1)} if quick else {
        "l2_grid": (200, 500, 1000, 2000)}
    log("Tuning EASE l2 + hybrid alpha/beta...")
    params, _ = tune(content, X, tune_split, k=k, log=log, **grids)
    params = {name: float(v) for name, v in params.items()}

    # -- stage 1 + 2 --------------------------------------------------------------
    collab = EASE(params["l2"]).fit(X_train).impute_cold(item_sim, IMPUTE_K, IMPUTE_POWER)
    stats = item_stats_from_ratings(ratings, user_ids[split.train_rows])
    params.update(impute_k=IMPUTE_K, impute_power=IMPUTE_POWER, n_collab=N_COLLAB, n_content=N_CONTENT)

    reranker = None
    if use_ranker:
        log("Training LightGBM reranker...")
        reranker = Reranker(content, collab, years=ratings.years, item_stats=stats,
                            n_collab=N_COLLAB, n_content=N_CONTENT)
        n_train = len(split.val) - min(RANKER_VALID_USERS, len(split.val) // 5)
        n_drop = int(n_train * DROPOUT_USERS)
        dropped = pick_cold_items(X_train, frac=DROPOUT_FRAC, seed=seed + 1000)
        dropout_ranker = Reranker(
            content, EASE(params["l2"]).fit(without_items(X_train, dropped)).impute_cold(
                item_sim, IMPUTE_K, IMPUTE_POWER),
            years=ratings.years, item_stats=_hide_stats(stats, dropped), n_collab=N_COLLAB, n_content=N_CONTENT)
        reranker.fit(_slice(split.val, slice(n_drop, n_train)),
                     _slice(split.val, slice(n_train, len(split.val))),
                     extra_train=[(dropout_ranker, _slice(split.val, slice(0, n_drop)))], log=log)
        params.update(dropout_users=DROPOUT_USERS, dropout_frac=DROPOUT_FRAC)
        params["candidate_recall"] = reranker.candidate_recall(split.test)

    # -- test: all movies ---------------------------------------------------------
    log("Evaluating on test users...")
    scorers = baseline_scorers(content, collab, params["alpha"], params["beta"])
    if reranker:
        scorers["two-stage (LightGBM)"] = lambda F, b: reranker.scores_batch(F)
    warm = {name: evaluate(fn, split.test, k) for name, fn in scorers.items()}

    # -- test: cold start ---------------------------------------------------------
    cold_items = pick_cold_items(X_train, seed=seed)
    collab_cold = EASE(params["l2"]).fit(without_items(X_train, cold_items)).impute_cold(
        item_sim, IMPUTE_K, IMPUTE_POWER)
    cold_users = cold_start_users(split.test, cold_items)
    hybrid_cold = HybridRecommender(content, collab_cold, params["alpha"], params["beta"])
    votes = np.log1p(df["votes"].to_numpy(float))
    cold_scorers = {
        "IMDb popularity": lambda F, b: votes,
        "content": lambda F, b: content.blend_quality(content.similarity_batch(F)),
        "hybrid (imputed collab)": lambda F, b: hybrid_cold.scores_batch(F),
    }
    if reranker:
        cold_ranker = Reranker(content, collab_cold, years=ratings.years, item_stats=_hide_stats(stats, cold_items),
                               booster=reranker.booster, n_collab=N_COLLAB, n_content=N_CONTENT)
        cold_scorers["two-stage (LightGBM)"] = lambda F, b: cold_ranker.scores_batch(F)
    cold = {name: evaluate_cold(fn, cold_users, cold_items, k) for name, fn in cold_scorers.items()}

    model = TrainedModel(content, collab, params, years=ratings.years, item_stats=stats,
                         booster=reranker.booster if reranker else None)
    info = {
        "dataset": dataset,
        "matched_movies": ratings.matched,
        "users": int(X.shape[0]),
        "likes": int(X.nnz),
        "split": {"train": len(split.train_rows), "ranker": len(split.val), "test": len(split.test),
                  "cold_items": len(cold_items), "cold_test_users": len(cold_users)},
        "seed": seed,
        "test_metrics": warm,
        "cold_start_metrics": cold,
        "feature_importance": reranker.feature_importance() if reranker else None,
    }
    return model, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=movielens.DEFAULT_DATASET, choices=sorted(movielens.DATASETS))
    parser.add_argument("--quick", action="store_true", help="small grids and user counts (smoke test)")
    parser.add_argument("--no-ranker", action="store_true", help="skip the LightGBM reranker")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    model, info = run(args.dataset, args.seed, args.quick, not args.no_ranker, args.k, log)
    print(f"\nTest users (n={info['split']['test']:,}), top-{args.k}, all movies:\n")
    print(format_table(info["test_metrics"]))
    print(f"\nCold start: {info['split']['cold_items']} movies hidden from collaborative training; "
          f"ranking them for {info['split']['cold_test_users']:,} test users who liked one:\n")
    print(format_table(info["cold_start_metrics"]))
    meta = model.save(extra=info)
    log(f"Saved {meta['model']} version {meta['version']} to models/")


if __name__ == "__main__":
    main()
