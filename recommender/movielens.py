"""MovieLens ratings: download, match to the IMDb catalog, build interactions.

MovieLens provides real user ratings; we keep only ratings of movies that
exist in our catalog and use them to train the collaborative model. The data
is downloaded on demand (GroupLens' license doesn't allow redistributing it).
"""
import io
import re
import ssl
import unicodedata
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .paths import MOVIELENS_DIR

DATASETS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}
DEFAULT_DATASET = "ml-latest-small"
POSITIVE_RATING = 4.0  # ratings >= this count as a "like"

_ARTICLES = ("the", "a", "an", "les", "la", "le", "il", "das", "der", "die", "el", "l'")


def _ssl_context():
    # python.org macOS builds often lack a CA bundle; prefer certifi's.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(dataset=DEFAULT_DATASET, dest=MOVIELENS_DIR, force=False):
    """Download and extract ratings.csv + movies.csv. Returns the folder."""
    out = dest / dataset
    if not force and (out / "ratings.csv").exists() and (out / "movies.csv").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASETS[dataset], timeout=120, context=_ssl_context()) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in ("ratings.csv", "movies.csv"):
            (out / name).write_bytes(zf.read(f"{dataset}/{name}"))
    return out


def load(dataset=DEFAULT_DATASET, dest=MOVIELENS_DIR):
    folder = download(dataset, dest)
    return pd.read_csv(folder / "movies.csv"), pd.read_csv(folder / "ratings.csv")


def title_key(title) -> str:
    """Matching key: ASCII, lowercase, alphanumeric, '&' -> 'and', no leading article."""
    t = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode().lower()
    t = t.replace("&", " and ").strip()
    for article in _ARTICLES:
        if t.startswith(article + " "):
            t = t[len(article) + 1:]
            break
    return re.sub(r"[^a-z0-9]", "", t)


def _movielens_title_keys(title):
    """Keys for a MovieLens title like "Lives of Others, The (Leben der Anderen, Das) (2006)".

    Returns (keys, year). Alternate titles in parentheses become extra keys,
    and trailing articles ("..., The") are moved to the front.
    """
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", str(title).strip())
    base, year = (m.group(1), int(m.group(2))) if m else (str(title), None)
    parts = [base] + re.findall(r"\(([^)]*)\)", base)
    keys = set()
    for part in parts:
        part = re.sub(r"\([^)]*\)", "", part).strip()
        part = re.sub(r"^a\.k\.a\.\s*", "", part)
        trailing = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|Das|Der|Die|El|L')$", part)
        if trailing:
            part = f"{trailing.group(2)} {trailing.group(1)}"
        if part:
            keys.add(title_key(part))
    return keys, year


def match_catalog(catalog_titles, ml_movies, ml_ratings):
    """Map each catalog row to a MovieLens movieId (or -1 when unmatched).

    The catalog has no release year, so when several MovieLens movies share a
    title we pick the most-rated one. Returns (movie_ids, years) arrays aligned
    with ``catalog_titles``; years come from MovieLens (-1 when unknown).
    """
    counts = ml_ratings.groupby("movieId").size()
    best = {}  # key -> (n_ratings, movieId, year)
    for movie_id, title in zip(ml_movies["movieId"], ml_movies["title"], strict=True):
        keys, year = _movielens_title_keys(title)
        cand = (int(counts.get(movie_id, 0)), int(movie_id), year if year else -1)
        for key in keys:
            if key not in best or cand > best[key]:
                best[key] = cand

    movie_ids = np.full(len(catalog_titles), -1, dtype=np.int64)
    years = np.full(len(catalog_titles), -1, dtype=np.int64)
    for i, title in enumerate(catalog_titles):
        hit = best.get(title_key(title))
        if hit:
            movie_ids[i], years[i] = hit[1], hit[2]
    return movie_ids, years


def interactions(ml_ratings, catalog_movie_ids, positive=POSITIVE_RATING):
    """Binary user x catalog-item matrix of positive ratings.

    Returns (X, user_ids) where X is a CSR matrix with one row per MovieLens
    user who liked at least one catalog movie, and columns are catalog ids.
    """
    col_of = {int(m): i for i, m in enumerate(catalog_movie_ids) if m >= 0}
    likes = ml_ratings[ml_ratings["rating"] >= positive]
    likes = likes[likes["movieId"].isin(col_of)]
    user_ids, rows = np.unique(likes["userId"].to_numpy(), return_inverse=True)
    cols = likes["movieId"].map(col_of).to_numpy()
    X = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(catalog_movie_ids)),
    )
    X.sum_duplicates()
    X.data[:] = 1.0
    return X, user_ids
