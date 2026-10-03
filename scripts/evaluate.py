#!/usr/bin/env python3
"""Robust offline comparison: rerun the full training pipeline on several
random user splits (each tuned and trained on its own users, no leakage) and
report mean ± std, then show example recommendations from the saved model.

Run from the project root (after `python -m scripts.train`):
    python -m scripts.evaluate               # 3 splits, ~4 min each on ML-32M
    python -m scripts.evaluate --splits 5
"""
import argparse
import json

from recommender.evaluation import aggregate, format_table
from recommender.model import TrainedModel
from recommender.paths import MODELS_DIR
from scripts.train import run

SANITY_TITLES = [["The Dark Knight"], ["Toy Story"], ["Amélie", "Before Sunrise"], ["Oppenheimer"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--dataset", default=None, help="defaults to the saved model's dataset")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    model = TrainedModel.load()
    dataset = args.dataset or model.metadata["dataset"]

    warm_runs, cold_runs = [], []
    for seed in range(args.splits):
        _, info = run(dataset, seed=seed, k=args.k, log=lambda *_: None)
        warm_runs.append(info["test_metrics"])
        cold_runs.append(info["cold_start_metrics"])
        print(f"  split {seed}: two-stage ndcg@{args.k} "
              f"{info['test_metrics']['two-stage (LightGBM)'][f'ndcg@{args.k}']:.4f} (all) / "
              f"{info['cold_start_metrics']['two-stage (LightGBM)'][f'ndcg@{args.k}']:.4f} (cold)", flush=True)

    warm, warm_std = aggregate(warm_runs)
    cold, cold_std = aggregate(cold_runs)
    print(f"\nAll movies, top-{args.k}, mean ± std over {args.splits} splits:\n")
    print(format_table(warm, warm_std))
    print(f"\nCold start, top-{args.k}, mean ± std over {args.splits} splits:\n")
    print(format_table(cold, cold_std))
    out = MODELS_DIR / "evaluation.json"
    out.write_text(json.dumps({"splits": args.splits, "dataset": dataset, "all_movies": {"mean": warm, "std": warm_std},
                               "cold_start": {"mean": cold, "std": cold_std}}, indent=2))
    print(f"\nWrote {out}")

    content = model.content
    print(f"\nExample recommendations (top 5) from saved model {model.metadata['version']}:")
    for titles in SANITY_TITLES:
        ids = [i for i in (content.find_title(t) for t in titles) if i is not None]
        if not ids:
            continue
        print(f"\n  liked: {', '.join(content.df.iloc[i]['title'] for i in ids)}")
        print(f"    content only: {', '.join(content.recommend_for(ids, k=5)['title'])}")
        print(f"    two-stage:    {', '.join(model.recommend_for(ids, k=5)['title'])}")


if __name__ == "__main__":
    main()
