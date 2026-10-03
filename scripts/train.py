#!/usr/bin/env python3
"""Train the hybrid recommender and save it to models/.

Steps:
  1. Download MovieLens (cached in data/movielens/) and match it to the catalog.
  2. Tune EASE l2 + hybrid alpha on validation users.
  3. Report all models on held-out test users.
  4. Refit EASE on every user and save models/hybrid.npz + hybrid.json
     (params, data stats and test metrics) for the API to load.

Run from the project root:
    python -m scripts.train
    python -m scripts.train --l2 200 --alpha 0.8 --beta 0.05   # skip tuning
"""
import argparse

import numpy as np

from recommender import movielens
from recommender.catalog import load_movie_df
from recommender.collaborative import EASE
from recommender.content import ContentRecommender
from recommender.evaluation import compare_on_test, format_table, split_users, tune
from recommender.hybrid import HybridRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=movielens.DEFAULT_DATASET, choices=sorted(movielens.DATASETS))
    parser.add_argument("--l2", type=float, help="EASE regularization (all three skip tuning)")
    parser.add_argument("--alpha", type=float, help="hybrid collab weight in [0,1]")
    parser.add_argument("--beta", type=float, help="hybrid quality-prior weight in [0,1]")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_movie_df()
    content = ContentRecommender(df)

    print(f"Loading MovieLens ({args.dataset})...")
    ml_movies, ml_ratings = movielens.load(args.dataset)
    movie_ids, years = movielens.match_catalog(df["title"], ml_movies, ml_ratings)
    X, _ = movielens.interactions(ml_ratings, movie_ids)
    matched = int((movie_ids >= 0).sum())
    print(f"Matched {matched}/{len(df)} catalog movies; {X.shape[0]} users, {X.nnz} likes.")

    split = split_users(X, seed=args.seed)
    print(f"Users: {len(split.train_rows)} train / {len(split.val)} val / {len(split.test)} test")

    if None not in (args.l2, args.alpha, args.beta):
        params = {"l2": args.l2, "alpha": args.alpha, "beta": args.beta}
    else:
        print("Tuning on validation users...")
        params, _ = tune(content, X, split, k=args.k)
    print(f"Selected {params}")

    results = compare_on_test(content, X, split, **params, k=args.k)
    print(f"\nTest users (n={len(split.test)}), top-{args.k}:\n")
    print(format_table(results))

    model = HybridRecommender(
        content, EASE(params["l2"]).fit(X), alpha=params["alpha"], beta=params["beta"], years=years
    )
    meta = model.save(extra={
        "dataset": args.dataset,
        "matched_movies": matched,
        "users": int(X.shape[0]),
        "likes": int(X.nnz),
        "seed": args.seed,
        "test_metrics": results,
    })
    print(f"\nSaved model version {meta['version']} to models/")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
