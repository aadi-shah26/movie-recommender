#!/usr/bin/env python3
"""Minimal downloader for the IMDB top-1000 Kaggle dataset.

This script downloads `mayankray/imdb-top-1000-movies-dataset` into ./data
and prints the first 5 rows of the first CSV found. No token handling.
"""
from pathlib import Path
import argparse
import kagglehub
import pandas as pd
import sys


DATASET = 'mayankray/imdb-top-1000-movies-dataset'
OUT_DIR = Path('data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Force re-download of the dataset')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # If CSV exists and refresh not asked for, skip download to avoid overwriting
    existing = list(OUT_DIR.rglob('*.csv'))
    if existing and not args.refresh:
        f = existing[0]
        print('Found existing CSV, skipping download (use --refresh to force):', f)
    else:
        print('Downloading dataset:', DATASET)
        try:
            kagglehub.dataset_download(DATASET, output_dir=str(OUT_DIR), force_download=args.refresh)
        except Exception as e:
            print('Download failed:', e)
            sys.exit(1)

        # find first CSV and print head
        files = list(OUT_DIR.rglob('*.csv'))
        if not files:
            print('No CSV files found under', OUT_DIR)
            sys.exit(1)
        f = files[0]

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
