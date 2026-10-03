"""The trained recommender as served by the API: load/save and inference.

    likes -> [content + EASE (with cold-start imputation)] -> candidates
          -> LightGBM LambdaRank reranker -> top-k

Ordering comes from the reranker. The score returned alongside each movie is
the hybrid blend in [0,1], which the UI shows as "% match" (LambdaRank scores
are only meaningful relative to each other), capped so it never increases
down the list.

Artifacts (models/, written by scripts/train.py):
    model.npz   EASE weights (with imputed cold-start rows/cols), co-like
                cosine, item counts/stats, MovieLens release years
    ranker.txt  LightGBM booster
    model.json  hyperparameters, data stats, offline metrics, version
"""
import json
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np

from .collaborative import EASE
from .content import ContentRecommender, likes_matrix, top_k
from .hybrid import HybridRecommender
from .paths import MODEL_ARTIFACT, MODEL_METADATA, RANKER_MODEL
from .ranker import Reranker


class TrainedModel:
    def __init__(self, content: ContentRecommender, collab: EASE, params, years=None, item_stats=None,
                 booster=None, metadata=None):
        if collab.B.shape[0] != content.n_items:
            raise ValueError(f"model has {collab.B.shape[0]} items but catalog has {content.n_items}; retrain")
        self.content, self.collab, self.params = content, collab, params
        self.years = years
        self.item_stats = item_stats or {}
        self.metadata = metadata or {}
        self.hybrid = HybridRecommender(content, collab, alpha=params["alpha"], beta=params["beta"])
        self.reranker = None
        if booster is not None:
            self.reranker = Reranker(content, collab, years=years, item_stats=item_stats, booster=booster,
                                     n_collab=params["n_collab"], n_content=params["n_content"])

    @property
    def name(self):
        return "two-stage-ease-lightgbm" if self.reranker else "hybrid-ease-content"

    def recommend_for(self, liked_ids, k=10):
        """DataFrame indexed by movie id with title + score (hybrid match in [0,1])."""
        liked = self.content.valid_ids(liked_ids)
        if not liked:
            return self.hybrid.popular(k)
        F = likes_matrix([liked], self.content.n_items)
        display = self.hybrid.scores_batch(F)[0]
        order = self.reranker.scores_batch(F)[0] if self.reranker else display
        idx = top_k(order, liked, k)
        shown = np.minimum.accumulate(np.clip(display[idx], 0.0, 1.0))
        return self.content.df.iloc[idx][["title"]].assign(score=shown)

    # ---- persistence ---------------------------------------------------------

    def save(self, models_dir=None, extra=None):
        artifact = MODEL_ARTIFACT if models_dir is None else models_dir / MODEL_ARTIFACT.name
        meta_path = MODEL_METADATA if models_dir is None else models_dir / MODEL_METADATA.name
        ranker_path = RANKER_MODEL if models_dir is None else models_dir / RANKER_MODEL.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        n = self.content.n_items
        np.savez_compressed(
            artifact, B=self.collab.B, cooc=self.collab.cooc, item_counts=self.collab.item_counts,
            imputed=self.collab.imputed, years=self.years if self.years is not None else np.full(n, -1),
            **{f"stat_{k}": v for k, v in self.item_stats.items()},
        )
        if self.reranker:
            booster = self.reranker.booster
            booster.save_model(str(ranker_path), num_iteration=booster.best_iteration or None)
        now = datetime.now(UTC)
        self.metadata = {
            "model": self.name,
            "version": now.strftime("%Y%m%d%H%M%S"),
            "trained_at": now.isoformat(timespec="seconds"),
            "n_items": n,
            "params": self.params,
            "content_weights": self.content.weights,
            **(extra or {}),
        }
        meta_path.write_text(json.dumps(self.metadata, indent=2, default=float))
        return self.metadata

    @classmethod
    def load(cls, df=None, models_dir=None):
        artifact = MODEL_ARTIFACT if models_dir is None else models_dir / MODEL_ARTIFACT.name
        meta_path = MODEL_METADATA if models_dir is None else models_dir / MODEL_METADATA.name
        ranker_path = RANKER_MODEL if models_dir is None else models_dir / RANKER_MODEL.name
        meta = json.loads(meta_path.read_text())
        content = ContentRecommender(df, weights=meta["content_weights"])
        collab = EASE(l2=meta["params"]["l2"])
        with np.load(artifact) as data:
            collab.B, collab.cooc = data["B"], data["cooc"]
            collab.item_counts, collab.imputed = data["item_counts"], data["imputed"]
            years = data["years"]
            stats = {k[len("stat_"):]: data[k] for k in data.files if k.startswith("stat_")}
        booster = lgb.Booster(model_file=str(ranker_path)) if ranker_path.exists() else None
        return cls(content, collab, meta["params"], years=years, item_stats=stats, booster=booster, metadata=meta)
