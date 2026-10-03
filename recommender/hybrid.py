"""Hybrid recommender: blends collaborative (EASE) and content similarity.

    score = (1 - beta) * [(1 - alpha) * content + alpha * collab] + beta * quality

``collab`` is the EASE score rescaled per request to [0,1]. Movies with no
MovieLens ratings (cold start, e.g. recent releases) have no collab score and
compete on content alone; if none of the user's likes have ratings, the blend
falls back to pure content. alpha and beta are tuned on held-out users
(see scripts/train.py).
"""
import json
from datetime import UTC, datetime

import numpy as np

from .collaborative import EASE
from .content import ContentRecommender, top_k
from .paths import HYBRID_ARTIFACT, HYBRID_METADATA


class HybridRecommender:
    def __init__(self, content: ContentRecommender, collab: EASE, alpha=0.8, beta=0.05,
                 years=None, metadata=None):
        if collab.B.shape[0] != content.n_items:
            raise ValueError(
                f"collab model has {collab.B.shape[0]} items but catalog has {content.n_items}; retrain"
            )
        self.content = content
        self.collab = collab
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.years = years
        self.metadata = metadata or {}

    def collab_scores(self, liked):
        """EASE scores rescaled to [0,1]; None when the likes carry no collab signal."""
        raw = self.collab.scores(liked)
        candidates = np.ones(len(raw), dtype=bool)
        candidates[liked] = False
        peak = raw[candidates & self.collab.covered].max(initial=0.0)
        if peak <= 0:
            return None
        return np.clip(raw / peak, 0.0, 1.0)

    def scores(self, liked_ids):
        liked = self.content.valid_ids(liked_ids)
        if not liked:
            return None
        sim = self.content.similarity(liked)
        cf = self.collab_scores(liked) if self.alpha else None
        mixed = sim if cf is None else (1 - self.alpha) * sim + self.alpha * cf
        return (1 - self.beta) * mixed + self.beta * self.content.quality

    def recommend_for(self, liked_ids, k=10):
        liked = self.content.valid_ids(liked_ids)
        scores = self.scores(liked)
        df = self.content.df
        if scores is None:
            # No likes yet: popular among MovieLens users, then by IMDb votes.
            pop = self.content.popularity_scores()
            pop = self.collab.item_counts * (pop.max() + 1) + pop
            idx = top_k(pop, [], k)
            return df.iloc[idx][["title"]].assign(score=0.0)
        idx = top_k(scores, liked, k)
        return df.iloc[idx][["title"]].assign(score=scores[idx])

    # ---- persistence -------------------------------------------------------

    def save(self, artifact=HYBRID_ARTIFACT, metadata_path=HYBRID_METADATA, extra=None):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            B=self.collab.B,
            item_counts=self.collab.item_counts,
            years=self.years if self.years is not None else np.full(self.content.n_items, -1),
        )
        meta = {
            "model": "hybrid-ease-content",
            "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_items": int(self.content.n_items),
            "l2": self.collab.l2,
            "alpha": self.alpha,
            "beta": self.beta,
            **(extra or {}),
        }
        metadata_path.write_text(json.dumps(meta, indent=2))
        self.metadata = meta
        return meta

    @classmethod
    def load(cls, content: ContentRecommender, artifact=HYBRID_ARTIFACT, metadata_path=HYBRID_METADATA):
        meta = json.loads(metadata_path.read_text())
        with np.load(artifact) as data:
            collab = EASE(l2=meta["l2"])
            collab.B = data["B"]
            collab.item_counts = data["item_counts"]
            years = data["years"]
        return cls(content, collab, alpha=meta["alpha"], beta=meta["beta"], years=years, metadata=meta)
