#!/usr/bin/env python3
"""Download the IMDb top-1000 Kaggle dataset (the movie catalog) into data/.

The CSV is already committed, so this is only needed to refresh it:
    python -m scripts.download_imdb --refresh
"""
import argparse
import sys

import kagglehub
import pandas as pd

from recommender.paths import CATALOG_CSV, DATA_DIR

DATASET = 'mayankray/imdb-top-1000-movies-dataset'
OUT_DIR = DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Force re-download of the dataset')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # If CSV exists and refresh not asked for, skip download to avoid overwriting
    if CATALOG_CSV.exists() and not args.refresh:
        f = CATALOG_CSV
        print('Found existing CSV, skipping download (use --refresh to force):', f)
    else:
        print('Downloading dataset:', DATASET)
        try:
            kagglehub.dataset_download(DATASET, output_dir=str(OUT_DIR), force_download=args.refresh)
        except Exception as e:
            print('Download failed:', e)
            sys.exit(1)

        if not CATALOG_CSV.exists():
            print('Expected CSV not found:', CATALOG_CSV)
            sys.exit(1)
        f = CATALOG_CSV

    print('Using CSV:', f)
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print('Failed to read CSV:', e)
        sys.exit(1)

    print('First 5 rows:')
    print(df.head())


if __name__ == '__main__':
    main()
