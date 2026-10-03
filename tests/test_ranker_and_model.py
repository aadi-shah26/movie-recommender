import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recommender.collaborative import EASE
from recommender.content import likes_matrix
from recommender.evaluation import evaluate, split_users
from recommender.model import TrainedModel
from recommender.ranker import FEATURES, Reranker, item_stats_from_ratings


def _synthetic_likes(n_items, n_users=600, seed=0):
    """Users who like random movies from one of a few "taste" clusters."""
    rng = np.random.default_rng(seed)
    clusters = np.array_split(rng.permutation(n_items)[:200], 5)
    rows, cols = [], []
    for u in range(n_users):
        items = rng.choice(clusters[u % 5], size=12, replace=False)
        rows += [u] * len(items)
        cols += items.tolist()
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items))


@pytest.fixture(scope="module")
def trained(content):
    X = _synthetic_likes(content.n_items)
    split = split_users(X, val_frac=0.3, test_frac=0.2, seed=0)
    collab = EASE(l2=10).fit(X[split.train_rows]).impute_cold(content.item_similarity(), k=5)
    reranker = Reranker(content, collab, n_collab=30, n_content=10)
    reranker.fit(split.val, split.test, rounds=20, log=lambda *_: None)
    return collab, reranker, split


def test_impute_cold_fills_only_unrated_items(content):
    X = _synthetic_likes(content.n_items)
    collab = EASE(l2=10).fit(X)
    before = collab.B.copy()
    collab.impute_cold(content.item_similarity(), k=5)
    assert collab.imputed.sum() == (~collab.covered).sum() > 0
    warm = np.ix_(collab.covered, collab.covered)
    assert np.array_equal(collab.B[warm], before[warm])
    assert np.abs(collab.B[:, collab.imputed]).sum() > 0  # cold movies now get collab scores
    assert collab.scored.all()


def test_reranker_features_and_scores(content, trained):
    collab, reranker, split = trained
    F = split.test.fold_in[:7]
    cand, X = reranker.candidates_and_features(F)
    assert cand.shape == (7, 40) and X.shape == (7 * 40, len(FEATURES))
    for u in range(7):  # candidates never include movies the user already liked
        assert not set(cand[u]) & set(F[u].indices)
        assert len(set(cand[u])) == 40
    scores = reranker.scores_batch(F)
    assert scores.shape == F.shape
    assert set(reranker.feature_importance()) == set(FEATURES)


def test_reranker_learns_clusters(content, trained):
    collab, reranker, split = trained
    m = evaluate(lambda F, b: reranker.scores_batch(F), split.test, k=10)
    pop = evaluate(lambda F, b: collab.item_counts, split.test, k=10)
    assert m["ndcg@10"] > pop["ndcg@10"]


def test_trained_model_roundtrip(tmp_path, content, trained):
    collab, reranker, _ = trained
    params = {"l2": 10.0, "alpha": 0.9, "beta": 0.0, "n_collab": 30, "n_content": 10}
    years = np.full(content.n_items, 1999)
    model = TrainedModel(content, collab, params, years=years, booster=reranker.booster)
    model.save(tmp_path, extra={"note": "test"})
    loaded = TrainedModel.load(content.df, models_dir=tmp_path)
    assert loaded.name == "two-stage-ease-lightgbm" and loaded.metadata["note"] == "test"
    liked = [0, 1, 2]
    a, b = model.recommend_for(liked, k=8), loaded.recommend_for(liked, k=8)
    assert a.index.tolist() == b.index.tolist()
    assert np.allclose(a["score"], b["score"])
    assert not set(liked) & set(a.index) and a["score"].between(0, 1).all()
    assert a["score"].is_monotonic_decreasing
    assert len(loaded.recommend_for([], k=4)) == 4


def test_item_stats_use_only_given_users():
    class R:
        users = np.array([1, 1, 2, 3])
        items = np.array([0, 1, 0, 1])
        ratings = np.array([5.0, 2.0, 1.0, 4.0], dtype=np.float32)
        n_items = 3

    stats = item_stats_from_ratings(R, user_ids=[1, 3])
    assert stats["ml_rating"][:2].tolist() == [5.0, 3.0]
    assert stats["like_rate"][:2].tolist() == [1.0, 0.5]
    assert np.isnan(stats["ml_rating"][2])


def test_likes_matrix():
    F = likes_matrix([[0, 2, 2], []], 4)
    assert F.toarray().tolist() == [[1, 0, 1, 0], [0, 0, 0, 0]]
