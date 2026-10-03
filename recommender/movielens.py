"""MovieLens ratings: download, match to the IMDb catalog, build interactions.

MovieLens provides real user ratings; we keep only ratings of movies that
exist in our catalog and use them to train the collaborative model. The data
is downloaded on demand (GroupLens' license doesn't allow redistributing it).

The 32M-rating dataset is too big to load whole on a laptop, so
``load_catalog_ratings`` streams it in chunks, keeps only ratings of movies
that could match the catalog, and caches that subset as a small .npz.
"""
import bisect
import hashlib
import re
import shutil
import ssl
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .paths import MOVIELENS_DIR

DATASETS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",  # 100k ratings
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",  # 32M ratings, 240 MB zip
}
DEFAULT_DATASET = "ml-32m"
CHUNK_ROWS = 4_000_000
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
    archive = dest / f"{dataset}.zip"
    if force or not archive.exists():
        print(f"Downloading {DATASETS[dataset]} ...")
        tmp = archive.with_suffix(".part")
        with urllib.request.urlopen(DATASETS[dataset], timeout=300, context=_ssl_context()) as resp, \
                open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.rename(archive)
    with zipfile.ZipFile(archive) as zf:
        for name in ("ratings.csv", "movies.csv"):
            with zf.open(f"{dataset}/{name}") as src, open(out / name, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    return out


def load(dataset=DEFAULT_DATASET, dest=MOVIELENS_DIR):
    """Full movies + ratings DataFrames. Fine for ml-latest-small; use
    ``load_catalog_ratings`` for the large datasets."""
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
    keys = {title_key(re.sub(r"[()]", "", base))}  # "(500) Days of Summer"
    for part in parts:
        part = re.sub(r"\([^)]*\)", "", part).strip()
        part = re.sub(r"^a\.k\.a\.\s*", "", part)
        trailing = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|Das|Der|Die|El|L')$", part)
        if trailing:
            part = f"{trailing.group(2)} {trailing.group(1)}"
        if part:
            keys.add(title_key(part))
    return keys, year


FUZZY_MIN_LEN = 5
FUZZY_MIN_RATIO = 0.3


class _TitleIndex:
    """MovieLens title keys, searchable exactly or by prefix/suffix.

    Exact key matches win. Otherwise a fuzzy match allows the catalog title to
    be a prefix or suffix of the MovieLens one ("Swades" / "Swades: We, the
    People", "Life of Brian" / "Monty Python's Life of Brian"), preferring
    prefixes, requiring the same digits (so "Guardians of the Galaxy Vol. 2"
    can't match the first film) and skipping documentaries ("... and His Tale
    of the Princess Kaguya"). The reverse direction (MovieLens title shorter)
    is not allowed: it matched "Avengers: Endgame" to "The Avengers".
    """

    def __init__(self, ml_movies):
        self.exact, fuzzy_keys = {}, set()
        genres = ml_movies["genres"] if "genres" in ml_movies else [""] * len(ml_movies)
        for movie_id, title, genre in zip(ml_movies["movieId"], ml_movies["title"], genres, strict=True):
            keys, year = _movielens_title_keys(title)
            for key in keys:
                if key:
                    self.exact.setdefault(key, []).append((int(movie_id), year if year else -1))
                    if "Documentary" not in str(genre):
                        fuzzy_keys.add(key)
        self.sorted_keys = sorted(fuzzy_keys)
        self.sorted_reversed = sorted(k[::-1] for k in fuzzy_keys)

    @staticmethod
    def _compatible(short, long_):
        return (len(short) / len(long_) >= FUZZY_MIN_RATIO
                and re.sub(r"\D", "", short) == re.sub(r"\D", "", long_))

    @staticmethod
    def _with_prefix(sorted_keys, prefix):
        i = bisect.bisect_left(sorted_keys, prefix)
        while i < len(sorted_keys) and sorted_keys[i].startswith(prefix):
            yield sorted_keys[i]
            i += 1

    def lookup(self, key):
        """([(movieId, year), ...], is_fuzzy): exact matches, else fuzzy ones."""
        if key in self.exact:
            return self.exact[key], False
        if len(key) < FUZZY_MIN_LEN:
            return [], False
        prefixed = self._with_prefix(self.sorted_keys, key)
        suffixed = (k[::-1] for k in self._with_prefix(self.sorted_reversed, key[::-1]))
        for found in (prefixed, suffixed):
            hits = [c for k in found if self._compatible(key, k) for c in self.exact[k]]
            if hits:
                return hits, True
        return [], False


def candidate_movie_ids(catalog_titles, ml_movies, index=None):
    """MovieLens movieIds whose title could match some catalog title."""
    index = index or _TitleIndex(ml_movies)
    return {movie_id for t in catalog_titles for movie_id, _ in index.lookup(title_key(t))[0]}


def match_catalog(catalog_titles, ml_movies, counts, index=None, report=None):
    """Map each catalog row to a MovieLens movieId (or -1 when unmatched).

    ``counts`` maps movieId -> number of ratings (a Series/dict, or a ratings
    DataFrame to count from). The catalog has no release year, so when several
    MovieLens movies match a title we pick the most-rated one. Returns
    (movie_ids, years) aligned with ``catalog_titles``; years from MovieLens.
    Pass a list as ``report`` to collect (catalog title, MovieLens title) for
    fuzzy matches, for eyeballing.
    """
    if isinstance(counts, pd.DataFrame):
        counts = counts.groupby("movieId").size()
    index = index or _TitleIndex(ml_movies)
    titles_by_id = dict(zip(ml_movies["movieId"], ml_movies["title"], strict=True)) if report is not None else {}

    movie_ids = np.full(len(catalog_titles), -1, dtype=np.int64)
    years = np.full(len(catalog_titles), -1, dtype=np.int64)
    used = set()
    for i, title in enumerate(catalog_titles):
        hits, fuzzy = index.lookup(title_key(title))
        # Duplicate catalog titles (two different films both called "Drishyam")
        # can't both be the same MovieLens movie: only the first one claims it.
        hits = [h for h in hits if h[0] not in used]
        if not hits:
            continue
        movie_id, year = max(hits, key=lambda h: (int(counts.get(h[0], 0)), h[0]))
        if fuzzy and not counts.get(movie_id, 0):
            continue
        movie_ids[i], years[i] = movie_id, year
        used.add(movie_id)
        if fuzzy and report is not None:
            report.append((str(title), titles_by_id[movie_id]))
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


@dataclass
class CatalogRatings:
    """MovieLens ratings restricted to catalog movies (item = catalog row id)."""
    users: np.ndarray       # MovieLens userId per rating
    items: np.ndarray       # catalog id per rating
    ratings: np.ndarray     # 0.5 .. 5.0
    movie_ids: np.ndarray   # catalog id -> MovieLens movieId (-1 = unmatched)
    years: np.ndarray       # catalog id -> release year from MovieLens (-1 = unknown)
    dataset: str

    @property
    def n_items(self):
        return len(self.movie_ids)

    @property
    def matched(self):
        return int((self.movie_ids >= 0).sum())

    def interactions(self, positive=POSITIVE_RATING):
        """Binary (users x catalog items) like-matrix + the MovieLens userIds per row."""
        keep = self.ratings >= positive
        user_ids, rows = np.unique(self.users[keep], return_inverse=True)
        X = csr_matrix(
            (np.ones(keep.sum(), dtype=np.float32), (rows, self.items[keep])),
            shape=(len(user_ids), self.n_items),
        )
        X.sum_duplicates()
        X.data[:] = 1.0
        return X, user_ids


def load_catalog_ratings(catalog_titles, dataset=DEFAULT_DATASET, dest=MOVIELENS_DIR) -> CatalogRatings:
    """Ratings of catalog movies only, streamed from disk and cached."""
    catalog_titles = [str(t) for t in catalog_titles]
    digest = hashlib.md5("\n".join(catalog_titles).encode()).hexdigest()[:10]
    cache = dest / dataset / f"catalog_ratings_{digest}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return CatalogRatings(d["users"], d["items"], d["ratings"], d["movie_ids"], d["years"], dataset)

    folder = download(dataset, dest)
    ml_movies = pd.read_csv(folder / "movies.csv")
    index = _TitleIndex(ml_movies)
    candidates = candidate_movie_ids(catalog_titles, ml_movies, index)

    parts, counts = [], pd.Series(dtype=np.int64)
    reader = pd.read_csv(
        folder / "ratings.csv", usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32}, chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk[chunk["movieId"].isin(candidates)]
        counts = counts.add(chunk.groupby("movieId").size(), fill_value=0)
        parts.append(chunk)
    ratings = pd.concat(parts, ignore_index=True)

    movie_ids, years = match_catalog(catalog_titles, ml_movies, counts, index)
    col_of = pd.Series(np.arange(len(movie_ids))[movie_ids >= 0], index=movie_ids[movie_ids >= 0])
    ratings = ratings[ratings["movieId"].isin(col_of.index)]
    result = CatalogRatings(
        users=ratings["userId"].to_numpy(np.int32),
        items=col_of.loc[ratings["movieId"]].to_numpy(np.int32),
        ratings=ratings["rating"].to_numpy(np.float32),
        movie_ids=movie_ids, years=years, dataset=dataset,
    )
    np.savez_compressed(cache, users=result.users, items=result.items, ratings=result.ratings,
                        movie_ids=movie_ids, years=years)
    return result
