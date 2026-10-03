import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from recommender.collaborative import EASE
from recommender.content import likes_matrix
from recommender.evaluation import EvalSet, evaluate, split_users
from recommender.hybrid import HybridRecommender
from recommender.model import TrainedModel


@pytest.fixture(scope="module")
def collab(content):
    X = sparse_random(300, content.n_items, density=0.02, format="csr", random_state=0)
    X.data[:] = 1
    return EASE(l2=50).fit(X)


def test_alpha_zero_is_content_only(content, collab):
    liked = [0, 1]
    hybrid = HybridRecommender(content, collab, alpha=0.0, beta=content.quality_weight)
    assert np.allclose(hybrid.scores(liked), content.scores(liked))


def test_hybrid_recommendations(content, collab):
    hybrid = HybridRecommender(content, collab, alpha=0.8, beta=0.05)
    recs = hybrid.recommend_for([0, 1, 2], k=10)
    assert len(recs) == 10 and not {0, 1, 2} & set(recs.index)
    assert recs["score"].between(0, 1).all()
    assert len(hybrid.recommend_for([], k=4)) == 4


def test_catalog_size_mismatch_is_rejected(content):
    small = EASE().fit(np.eye(3))
    with pytest.raises(ValueError, match="retrain"):
        HybridRecommender(content, small)
    with pytest.raises(ValueError, match="retrain"):
        TrainedModel(content, small, {"alpha": 0.9, "beta": 0.0})


def test_evaluate_metrics():
    users = EvalSet(np.array([0]), likes_matrix([[0]], 5), likes_matrix([[1, 2]], 5))
    perfect = evaluate(lambda F, b: np.array([9.0, 8, 7, 0, 0]), users, k=2)
    assert perfect["recall@2"] == 1.0 and perfect["ndcg@2"] == pytest.approx(1.0)
    miss = evaluate(lambda F, b: np.array([0, 0, 0, 9.0, 8]), users, k=2)
    assert miss["recall@2"] == 0.0 and miss["coverage"] == 0.4


def test_split_users_is_disjoint_and_hides_likes():
    X = sparse_random(200, 50, density=0.2, format="csr", random_state=1)
    X.data[:] = 1
    split = split_users(X, min_likes=5, seed=0)
    groups = [set(split.train_rows), set(split.val.rows), set(split.test.rows)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for users in (split.val, split.test):
        assert (users.fold_in.multiply(users.held_out)).nnz == 0
        assert (np.asarray(users.held_out.sum(axis=1)) >= 1).all()
        assert ((users.fold_in + users.held_out) != X[users.rows]).nnz == 0


def test_batch_scores_match_single_user(content, collab):
    hybrid = HybridRecommender(content, collab, alpha=0.6, beta=0.1)
    liked = [[0, 5, 9], [3], [100, 200]]
    batch = hybrid.scores_batch(likes_matrix(liked, content.n_items))
    for row, ids in zip(batch, liked, strict=True):
        assert np.allclose(row, hybrid.scores(ids), atol=1e-6)
