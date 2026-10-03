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

    def fit(self, X):
        """X: (users x items) binary interaction matrix, sparse or dense."""
        G = np.asarray((X.T @ X).todense() if hasattr(X, "todense") else X.T @ X, dtype=np.float64)
        self.item_counts = np.diag(G).copy()
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    @property
    def covered(self):
        """Items seen in training; others have no collaborative signal."""
        return self.item_counts > 0

    def scores(self, liked):
        """Score every item for a user who liked the item ids in ``liked``."""
        if not len(liked):
            return np.zeros(self.B.shape[0], dtype=np.float32)
        return self.B[np.asarray(liked)].sum(axis=0)
