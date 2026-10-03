"""Hybrid recommender: blends collaborative (EASE) and content similarity.

    score = (1 - beta) * [(1 - alpha) * content + alpha * collab] + beta * quality

``collab`` is the EASE score rescaled per request to [0,1]. Movies with no
MovieLens ratings (cold start, e.g. recent releases) have no collab score and
compete on content alone; if none of the user's likes have ratings, the blend
falls back to pure content. alpha and beta are tuned on held-out users
(see scripts/train.py).

Movies with no ratings get imputed collab scores (EASE.impute_cold), so they
compete on equal footing instead of having only the content share.
"""
import numpy as np

from .collaborative import EASE
from .content import ContentRecommender, likes_matrix, top_k


def rescale_collab(cf_raw, F, covered):
    """Rescale each row of raw EASE scores to [0,1] by its best candidate.

    Candidates are items the user hasn't liked that have collab signal. Rows
    whose likes carry no collab signal come back as NaN.
    """
    masked = np.where(covered, cf_raw, -np.inf)
    rows, cols = F.nonzero()
    masked[rows, cols] = -np.inf
    peak = masked.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cf = np.clip(cf_raw / peak, 0.0, 1.0)
    cf[(peak <= 0).ravel()] = np.nan
    return cf


def blend(sim, cf_raw, F, covered, quality, alpha, beta):
    """(1-beta) * [(1-alpha) * content + alpha * collab] + beta * quality, row-wise.

    Users with no collab signal get pure content in place of the mix.
    """
    if alpha:
        cf = rescale_collab(cf_raw, F, covered)
        mixed = np.where(np.isnan(cf), sim, (1 - alpha) * sim + alpha * np.nan_to_num(cf))
    else:
        mixed = sim
    return (1 - beta) * mixed + beta * quality


class HybridRecommender:
    def __init__(self, content: ContentRecommender, collab: EASE, alpha=0.9, beta=0.0):
        if collab.B.shape[0] != content.n_items:
            raise ValueError(
                f"collab model has {collab.B.shape[0]} items but catalog has {content.n_items}; retrain"
            )
        self.content = content
        self.collab = collab
        self.alpha = float(alpha)
        self.beta = float(beta)

    def scores(self, liked_ids):
        liked = self.content.valid_ids(liked_ids)
        if not liked:
            return None
        return self.scores_batch(likes_matrix([liked], self.content.n_items))[0]

    def scores_batch(self, F):
        """Hybrid scores for a sparse (users x items) like-matrix."""
        return blend(
            self.content.similarity_batch(F), self.collab.scores_batch(F), F,
            self.collab.scored, self.content.quality, self.alpha, self.beta,
        )

    def popular(self, k):
        """No likes yet: popular among MovieLens users, then by IMDb votes."""
        pop = self.content.popularity_scores()
        idx = top_k(self.collab.item_counts * (pop.max() + 1) + pop, [], k)
        return self.content.df.iloc[idx][["title"]].assign(score=0.0)

    def recommend_for(self, liked_ids, k=10):
        liked = self.content.valid_ids(liked_ids)
        scores = self.scores(liked)
        if scores is None:
            return self.popular(k)
        idx = top_k(scores, liked, k)
        return self.content.df.iloc[idx][["title"]].assign(score=scores[idx])
