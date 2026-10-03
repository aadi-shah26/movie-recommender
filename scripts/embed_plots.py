#!/usr/bin/env python3
"""Precompute sentence embeddings of each movie's plot for the content model.

Run once (needs the extra deps in requirements-dev.txt: sentence-transformers):
    python -m scripts.embed_plots
    python -m scripts.embed_plots --model sentence-transformers/all-mpnet-base-v2

Writes data/plot_embeddings.npy (n_movies x dim, L2-normalized) plus a small
JSON sidecar. The API and training only read the .npy, so they never need
PyTorch at runtime.
"""
import argparse
import json

import numpy as np

from recommender.catalog import load_movie_df
from recommender.paths import PLOT_EMBEDDINGS, PLOT_EMBEDDINGS_META

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def movie_text(row):
    plot = row.get("Plot") if isinstance(row.get("Plot"), str) else ""
    return f"{row['title']}. {row['genres']}. {plot}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    df = load_movie_df()
    texts = [movie_text(row) for _, row in df.iterrows()]
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(PLOT_EMBEDDINGS, emb.astype(np.float32))
    PLOT_EMBEDDINGS_META.write_text(json.dumps({"model": args.model, "n": len(texts), "dim": emb.shape[1]}, indent=2))
    print(f"Wrote {PLOT_EMBEDDINGS} {emb.shape}")


if __name__ == "__main__":
    main()
