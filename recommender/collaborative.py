"""Collaborative filtering with EASE (Steck, "Embarrassingly Shallow
Autoencoders for Sparse Data", WWW 2019).

EASE learns an item-item weight matrix B by solving

    min_B ||X - XB||^2 + lambda * ||B||^2   s.t.  diag(B) = 0

which has the closed form  P = (X^T X + lambda I)^-1,  B_ij = -P_ij / P_jj.
Scoring a user is a single product: scores = x_user @ B. The catalog is small
(~1k items), so training is one dense inverse and takes well under a second.
"""
import numpy as np


class EASE:
    def __init__(self, l2=200.0):
        self.l2 = float(l2)
        self.B = None
        self.item_counts = None
        self.cooc = None
        self.imputed = None

    def fit(self, X):
        """X: (users x items) binary interaction matrix, sparse or dense."""
        G = np.asarray((X.T @ X).todense() if hasattr(X, "todense") else X.T @ X, dtype=np.float64)
        self.item_counts = np.diag(G).copy()
        # Co-like cosine similarity (how often two movies are liked by the same
        # users, normalized by popularity); a reranker feature.
        norms = np.sqrt(np.maximum(self.item_counts, 1.0))
        self.cooc = (G / norms[:, None] / norms[None, :]).astype(np.float32)
        np.fill_diagonal(self.cooc, 0.0)
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.imputed = np.zeros(len(B), dtype=bool)
        return self

    def impute_cold(self, item_sim, k=25, power=4.0):
        """Give items with no ratings (cold start) borrowed collaborative scores.

        Each uncovered item's column (and row) of B becomes a weighted average of the
        columns of its ``k`` most content-similar covered items, weighted by
        similarity**power. Scoring is linear in B, so a user's score for the
        cold item is the same weighted average of their scores for those
        neighbors: "people who'd like this new movie are the people who like
        the movies most like it". Tuned on simulated cold items (see
        scripts/train.py); far better than content similarity alone.
        """
        sim = np.array(item_sim, dtype=np.float32, copy=True)
        sim[:, ~self.covered] = -np.inf
        np.fill_diagonal(sim, -np.inf)
        cold = np.flatnonzero(~self.covered)
        k = min(k, int(self.covered.sum()))
        weights = {}
        for i in cold:
            nb = np.argpartition(-sim[i], k - 1)[:k]
            w = np.maximum(sim[i, nb], 0.0) ** power
            if w.sum() > 0:
                weights[i] = (nb, w / w.sum())
        # Columns: how much each liked movie points to the cold movie.
        for i, (nb, w) in weights.items():
            self.B[:, i] = self.B[:, nb] @ w
        # Rows: liking a cold movie counts like liking its neighbors.
        for i, (nb, w) in weights.items():
            self.B[i, :] = w @ self.B[nb, :]
            self.B[i, i] = 0.0
            self.imputed[i] = True
        return self

    @property
    def scored(self):
        """Items with a collaborative score: rated in training, or imputed."""
        return self.covered | self.imputed

    @property
    def covered(self):
        """Items seen in training; others have no collaborative signal."""
        return self.item_counts > 0

    def scores(self, liked):
        """Score every item for a user who liked the item ids in ``liked``."""
        if not len(liked):
            return np.zeros(self.B.shape[0], dtype=np.float32)
        return self.B[np.asarray(liked)].sum(axis=0)

    def scores_batch(self, F):
        """(users x items) scores for a sparse (users x items) like-matrix."""
        return np.asarray(F @ self.B, dtype=np.float32)
