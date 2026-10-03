"""Content-based recommender: cosine similarity over item metadata features."""
import difflib
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from .catalog import load_movie_df, normalize_key

DEFAULT_WEIGHTS = {
    "text": 1.0, "genre": 6.0, "director": 3.0, "actors": 4.0, "year": 2.0,
    # blend weight for the rating/popularity prior applied at ranking time
    "quality": 0.15,
}


def _name_tokenizer(s):
    """Tokenize a names field into whole-name tokens.

    Splits on '|' and ',' so multi-name fields break apart, but a single
    name like "Christopher Nolan" stays one token (lowercased), so two people
    who merely share a first/last name don't spuriously match.
    """
    if not s:
        return []
    return [p.strip().lower() for p in re.split(r"[|,]", str(s)) if p.strip()]


def _minmax(values, n):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    if s.notna().sum() == 0:
        return np.zeros(n)
    s = s.fillna(s.median())
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return np.zeros(n)
    return ((s - lo) / (hi - lo)).to_numpy()


def top_k(scores, exclude_ids, k):
    """Indices of the k highest scores, skipping ``exclude_ids``."""
    scores = np.asarray(scores, dtype=float).copy()
    exclude = [i for i in exclude_ids if 0 <= i < len(scores)]
    scores[exclude] = -np.inf
    k = min(k, len(scores) - len(set(exclude)))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class ContentRecommender:
    def __init__(self, df=None, weights=None, normalize_features=True):
        self.df = df if df is not None else load_movie_df()
        self.normalize_features = normalize_features
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.quality_weight = float(w["quality"])
        n = len(self.df)

        def _names_block(series):
            series = series.fillna("").astype(str).str.strip()
            if not series.str.len().gt(0).any():
                return csr_matrix((n, 0))
            vec = CountVectorizer(tokenizer=_name_tokenizer, token_pattern=None, lowercase=False)
            return vec.fit_transform(series)

        text_mat = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform(
            self.df["text"].fillna("")
        )

        genres = self.df["genres"].fillna("").astype(str).str.strip()
        if genres.str.len().gt(0).any():
            genre_mat = CountVectorizer(token_pattern=r"[^,|\s]+").fit_transform(genres)
        else:
            genre_mat = csr_matrix((n, 0))

        director_mat = _names_block(self.df["director"])
        actor_mat = _names_block(self.df["actors"])

        # Year scaled into [0,1]; a 0-column block when no usable year exists.
        year_mat = csr_matrix((n, 0))
        if self.df["year"].notna().any():
            years = self.df["year"].fillna(self.df["year"].median())
            if years.max() > years.min():
                scaled = (years - years.min()) / (years.max() - years.min())
                year_mat = csr_matrix(scaled.astype(float).to_numpy().reshape(-1, 1))

        # Per-block L2 normalization so weights mean relative importance rather
        # than being swamped by token counts, then weight.
        blocks = [
            (text_mat, w["text"]),
            (genre_mat, w["genre"]),
            (director_mat, w["director"]),
            (actor_mat, w["actors"]),
            (year_mat, w["year"]),
        ]
        scaled_blocks = []
        for block, weight in blocks:
            if block.shape[1] and self.normalize_features:
                block = normalize(block, norm="l2", axis=1)
            scaled_blocks.append(block.multiply(weight) if block.shape[1] else block)

        # L2-normalize rows so linear_kernel == cosine similarity.
        self.matrix = hstack(scaled_blocks, format="csr")
        if self.normalize_features:
            self.matrix = normalize(self.matrix, norm="l2", axis=1)

        # Quality/popularity prior in [0,1] from rating + log(votes).
        rating_q = _minmax(self.df["rating"], n)
        votes_q = _minmax(np.log1p(pd.to_numeric(self.df["votes"], errors="coerce")), n)
        self.quality = 0.5 * rating_q + 0.5 * votes_q

        self.title_to_idx = {normalize_key(t): i for i, t in enumerate(self.df["title"].astype(str))}

    @property
    def n_items(self):
        return self.matrix.shape[0]

    def valid_ids(self, liked_ids):
        """Drop invalid ids and duplicates, preserving order."""
        return list(dict.fromkeys(
            int(i) for i in liked_ids
            if isinstance(i, (int, np.integer)) and 0 <= i < self.n_items
        ))

    def find_title(self, title):
        key = normalize_key(title)
        if key in self.title_to_idx:
            return self.title_to_idx[key]
        matches = difflib.get_close_matches(key, list(self.title_to_idx), n=1, cutoff=0.5)
        return self.title_to_idx[matches[0]] if matches else None

    def similarity(self, liked):
        """Cosine similarity of every item to the mean profile of ``liked``."""
        profile = np.asarray(self.matrix[liked].mean(axis=0)).reshape(1, -1)
        if self.normalize_features:
            profile = normalize(profile, norm="l2", axis=1)
        return linear_kernel(profile, self.matrix).ravel()

    def blend_quality(self, scores):
        """Convex blend with the quality prior; keeps scores in [0,1]."""
        beta = self.quality_weight
        return (1 - beta) * scores + beta * self.quality if beta else scores

    def popularity_scores(self):
        votes = pd.to_numeric(self.df["votes"], errors="coerce")
        if votes.notna().any():
            return votes.fillna(0).to_numpy(dtype=float)
        return pd.to_numeric(self.df["rating"], errors="coerce").fillna(0).to_numpy(dtype=float)

    def scores(self, liked_ids):
        liked = self.valid_ids(liked_ids)
        if not liked:
            return None
        return self.blend_quality(self.similarity(liked))

    def recommend_for(self, liked_ids, k=10):
        """Stateless top-k: returns a DataFrame indexed by movie id with a score.

        With no valid likes, falls back to the most popular titles (score 0).
        """
        liked = self.valid_ids(liked_ids)
        scores = self.scores(liked)
        if scores is None:
            idx = top_k(self.popularity_scores(), [], k)
            return self.df.iloc[idx][["title"]].assign(score=0.0)
        idx = top_k(scores, liked, k)
        return self.df.iloc[idx][["title"]].assign(score=scores[idx])
