#!/usr/bin/env python3
"""Robust offline comparison: repeat tune-then-test over several random user
splits and report mean ± std, then print example recommendations from the
saved model for eyeball sanity-checking.

A single split has only ~80 test users, so one run is noisy; averaging over
splits (each tuned on its own validation users, no leakage) is what the
headline numbers should come from.

Run from the project root (after `python -m scripts.train`):
    python -m scripts.evaluate
    python -m scripts.evaluate --splits 10
"""
import argparse

from recommender import movielens
from recommender.catalog import load_movie_df
from recommender.content import ContentRecommender
from recommender.evaluation import aggregate, compare_on_test, format_table, split_users, tune
from recommender.hybrid import HybridRecommender

SANITY_TITLES = [["The Dark Knight"], ["The Godfather"], ["Toy Story"], ["Amélie", "Before Sunrise"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5, help="number of random user splits")
    args = parser.parse_args()

    df = load_movie_df()
    content = ContentRecommender(df)
    model = HybridRecommender.load(content)
    meta = model.metadata

    ml_movies, ml_ratings = movielens.load(meta["dataset"])
    movie_ids, _ = movielens.match_catalog(df["title"], ml_movies, ml_ratings)
    X, _ = movielens.interactions(ml_ratings, movie_ids)

    runs = []
    for seed in range(args.splits):
        split = split_users(X, seed=seed)
        params, _ = tune(content, X, split, k=args.k, log=lambda *_: None)
        runs.append(compare_on_test(content, X, split, **params, k=args.k))
        print(f"  split {seed}: {params}  hybrid ndcg@{args.k}={runs[-1]['hybrid'][f'ndcg@{args.k}']:.4f}")

    mean, std = aggregate(runs)
    print(f"\nTest users, top-{args.k}, mean ± std over {args.splits} splits:\n")
    print(format_table(mean, std))

    print(f"\nExample recommendations (top 5) from saved model {meta['version']}: content vs hybrid")
    for titles in SANITY_TITLES:
        ids = [i for i in (content.find_title(t) for t in titles) if i is not None]
        if not ids:
            continue
        print(f"\n  liked: {', '.join(df.iloc[i]['title'] for i in ids)}")
        c = content.recommend_for(ids, k=5)["title"].tolist()
        h = model.recommend_for(ids, k=5)["title"].tolist()
        for a, b in zip(c, h, strict=True):
            print(f"    {a[:38]:<40}{b[:38]}")


if __name__ == "__main__":
    main()
