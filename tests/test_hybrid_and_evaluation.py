import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from recommender.collaborative import EASE
from recommender.evaluation import EvalUser, evaluate, split_users
from recommender.hybrid import HybridRecommender


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


def test_save_load_roundtrip(tmp_path, content, collab):
    years = np.arange(content.n_items)
    hybrid = HybridRecommender(content, collab, alpha=0.7, beta=0.1, years=years)
    hybrid.save(tmp_path / "m.npz", tmp_path / "m.json", extra={"note": "test"})
    loaded = HybridRecommender.load(content, tmp_path / "m.npz", tmp_path / "m.json")
    assert (loaded.alpha, loaded.beta, loaded.metadata["note"]) == (0.7, 0.1, "test")
    assert np.allclose(loaded.scores([3, 4]), hybrid.scores([3, 4]))
    assert loaded.years.tolist() == years.tolist()


def test_catalog_size_mismatch_is_rejected(content):
    small = EASE().fit(np.eye(3))
    with pytest.raises(ValueError, match="retrain"):
        HybridRecommender(content, small)


def test_evaluate_metrics():
    users = [EvalUser(fold_in=np.array([0]), held_out=np.array([1, 2]))]
    perfect = evaluate(lambda liked: np.array([9.0, 8, 7, 0, 0]), users, n_items=5, k=2)
    assert perfect["recall@2"] == 1.0 and perfect["ndcg@2"] == pytest.approx(1.0)
    miss = evaluate(lambda liked: np.array([0, 0, 0, 9.0, 8]), users, n_items=5, k=2)
    assert miss["recall@2"] == 0.0 and miss["coverage"] == 0.4


def test_split_users_is_disjoint_and_hides_likes(collab):
    X = sparse_random(200, 50, density=0.2, format="csr", random_state=1)
    X.data[:] = 1
    split = split_users(X, min_likes=5, seed=0)
    groups = [set(split.train_rows), set(split.val_rows), set(split.test_rows)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for user in split.test:
        assert len(user.held_out) >= 1 and not set(user.fold_in) & set(user.held_out)
