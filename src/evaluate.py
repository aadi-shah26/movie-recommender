#!/usr/bin/env python3
"""Offline evaluation harness for the content-based recommender.

We have no ground-truth user interactions, so quality is measured with
content-proxy metrics: given a "basket" of liked movies, do the top-k
recommendations actually share genres / director / cast with the seeds, and
are they reasonably well-regarded films?

Metrics (averaged over all baskets, higher = better except where noted):
  - genre_jaccard : mean Jaccard overlap of each rec's genres vs the union of
                    the basket's genres
  - director_rate : fraction of recs sharing a director with the basket
  - actor_overlap : fraction of recs sharing >=1 cast member with the basket
  - mean_rating   : mean IMDb rating of the recommended films

Run:
    python src/evaluate.py            # evaluate the current (new) config
    python src/evaluate.py --compare  # baseline (old) vs new, side by side
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from personal_recommender import load_movie_df, PersonalRecommender

# Repromducible sampling without Math.random-style nondeterminism.
RNG_SEED = 0
N_RANDOM_BASKETS = 40
BASKET_SIZE = 3
TOP_K = 10

# A few fixed, recognizable baskets for eyeball sanity-checking.
SANITY_TITLES = [
    ["The Dark Knight"],
    ["The Godfather"],
    ["Toy Story"],
]


def _genre_set(val):
    return {g.strip().lower() for g in str(val).split(",") if g.strip()}


def _name_set(val):
    # actors are pipe-joined; director is a plain name
    parts = str(val).replace("|", ",").split(",")
    return {p.strip().lower() for p in parts if p.strip()}


def _basket_signature(df, ids):
    genres, directors, actors = set(), set(), set()
    for i in ids:
        row = df.iloc[i]
        genres |= _genre_set(row.get("genres", ""))
        directors |= _name_set(row.get("director", ""))
        actors |= _name_set(row.get("actors", ""))
    return genres, directors, actors


def evaluate(rec, df, baskets, k=TOP_K):
    agg = {"genre_jaccard": [], "director_rate": [], "actor_overlap": [], "mean_rating": []}
    for ids in baskets:
        seed_genres, seed_dirs, seed_actors = _basket_signature(df, ids)
        recs = rec.recommend_for(list(ids), k=k)
        rec_ids = list(recs.index)
        if not rec_ids:
            continue
        g_scores, d_hits, a_hits, ratings = [], 0, 0, []
        for ri in rec_ids:
            row = df.iloc[ri]
            rg = _genre_set(row.get("genres", ""))
            union = seed_genres | rg
            g_scores.append(len(seed_genres & rg) / len(union) if union else 0.0)
            if _name_set(row.get("director", "")) & seed_dirs:
                d_hits += 1
            if _name_set(row.get("actors", "")) & seed_actors:
                a_hits += 1
            r = row.get("rating", np.nan)
            if r == r:  # not NaN
                ratings.append(float(r))
        agg["genre_jaccard"].append(np.mean(g_scores))
        agg["director_rate"].append(d_hits / len(rec_ids))
        agg["actor_overlap"].append(a_hits / len(rec_ids))
        if ratings:
            agg["mean_rating"].append(np.mean(ratings))
    return {m: (float(np.mean(v)) if v else 0.0) for m, v in agg.items()}


def _build_baskets(df):
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)
    baskets = [tuple(int(x) for x in rng.choice(n, size=BASKET_SIZE, replace=False))
               for _ in range(N_RANDOM_BASKETS)]
    # add fixed sanity baskets resolved by title
    tmp = PersonalRecommender(df)
    for titles in SANITY_TITLES:
        ids = [tmp._find_title(t) for t in titles]
        ids = [i for i in ids if i is not None]
        if ids:
            baskets.append(tuple(ids))
    return baskets


def _print_report(label, metrics):
    print(f"\n=== {label} ===")
    for m, v in metrics.items():
        print(f"  {m:<14} {v:.4f}")


def _print_sanity(rec, df):
    print("\n=== sanity baskets (top 5) ===")
    for titles in SANITY_TITLES:
        ids = [rec._find_title(t) for t in titles]
        ids = [i for i in ids if i is not None]
        if not ids:
            continue
        recs = rec.recommend_for(ids, k=5)
        print(f"\n  liked: {titles}")
        for ri, sc in zip(recs.index, recs.score):
            print(f"    {sc:.3f}  {df.iloc[ri]['title']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="Compare the old baseline config vs the new config")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    df = load_movie_df()
    baskets = _build_baskets(df)
    print(f"Evaluating on {len(baskets)} baskets (size {BASKET_SIZE}), top-{args.k}.")

    new_rec = PersonalRecommender(df)
    new_metrics = evaluate(new_rec, df, baskets, k=args.k)

    if args.compare:
        # Baseline approximates the original model: raw dot product (no
        # normalization), no actor signal, no quality blend.
        base_rec = PersonalRecommender(
            df, weights={"actors": 0.0, "quality": 0.0}, normalize_features=False
        )
        base_metrics = evaluate(base_rec, df, baskets, k=args.k)
        _print_report("BASELINE (old: no-norm, no-actors, no-quality)", base_metrics)
        _print_report("NEW (cosine + actors + quality)", new_metrics)
        print("\n=== delta (new - baseline) ===")
        for m in new_metrics:
            print(f"  {m:<14} {new_metrics[m] - base_metrics[m]:+.4f}")
    else:
        _print_report("NEW (cosine + actors + quality)", new_metrics)

    _print_sanity(new_rec, df)


if __name__ == "__main__":
    main()
