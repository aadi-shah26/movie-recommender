#!/usr/bin/env python3
"""One-time backfill: fetch a TMDB poster URL for every movie title.

Reads the movie catalog (via load_movie_df), queries the TMDB search API for
each title, and writes a {title: poster_url} map to data/posters.json. The
backend loads that file at startup and serves a `poster_url` per movie; the
frontend falls back to its generated tile for any title that didn't match.

This is intentionally a separate offline step so the API is hit ONCE, not at
runtime. Re-running resumes: titles already in posters.json are skipped, so an
interrupted run (or newly added movies) can be topped up cheaply.

Auth (set ONE of these env vars):
  TMDB_BEARER   - v4 "API Read Access Token" (recommended; Bearer header)
  TMDB_API_KEY  - v3 API key (sent as ?api_key=)

Usage:
  export TMDB_BEARER="eyJhbGci..."     # or TMDB_API_KEY="..."
  python -m scripts.fetch_posters
  python -m scripts.fetch_posters --refresh   # ignore cache, refetch everything
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

from recommender.catalog import load_movie_df
from recommender.paths import POSTERS_JSON

# macOS python.org builds often lack a CA bundle, so urllib's HTTPS verification
# fails ("CERTIFICATE_VERIFY_FAILED"). Use certifi's bundle when available.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
OUT_PATH = POSTERS_JSON
SAVE_EVERY = 25
REQUEST_PAUSE = 0.05  # be polite; TMDB allows ~50 req/s


def _build_request(title):
    bearer = os.environ.get("TMDB_BEARER")
    api_key = os.environ.get("TMDB_API_KEY")
    params = {"query": title, "include_adult": "false"}
    headers = {"User-Agent": "movie-recommender/1.0", "Accept": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    elif api_key:
        params["api_key"] = api_key
    else:
        sys.exit("ERROR: set TMDB_BEARER or TMDB_API_KEY in the environment.")
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    return urllib.request.Request(url, headers=headers)


def _poster_for(title, retries=2):
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(_build_request(title), timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
            results = data.get("results") or []
            for r in results:
                if r.get("poster_path"):
                    return IMG_BASE + r["poster_path"]
            return None  # matched search but no poster art
        except Exception as e:
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            print(f"  ! {title!r}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refetch all, ignore cache")
    args = parser.parse_args()

    df = load_movie_df()
    titles = list(dict.fromkeys(df["title"].astype(str).tolist()))

    posters = {}
    if OUT_PATH.exists() and not args.refresh:
        posters = json.loads(OUT_PATH.read_text())
        print(f"Resuming: {len(posters)} already cached.")

    todo = [t for t in titles if t not in posters]
    print(f"Fetching posters for {len(todo)} / {len(titles)} titles...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    found = sum(1 for v in posters.values() if v)
    for i, title in enumerate(todo, 1):
        url = _poster_for(title)
        posters[title] = url
        found += 1 if url else 0
        if i % SAVE_EVERY == 0:
            OUT_PATH.write_text(json.dumps(posters, ensure_ascii=False, indent=0))
            print(f"  {i}/{len(todo)} done ({found} with art)")
        time.sleep(REQUEST_PAUSE)

    OUT_PATH.write_text(json.dumps(posters, ensure_ascii=False, indent=0))
    hit_rate = found / len(titles) * 100 if titles else 0
    print(f"\nDone. {found}/{len(titles)} titles have posters ({hit_rate:.0f}%).")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
