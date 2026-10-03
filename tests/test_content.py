import numpy as np

from recommender.content import top_k


def test_catalog_shape(catalog):
    assert len(catalog) == 1000
    assert {"title", "genres", "director", "actors", "rating", "votes"} <= set(catalog.columns)
    assert catalog.loc[0, "title"] == "The Shawshank Redemption"
    assert "|" in catalog.loc[0, "actors"]  # list-literal cast parsed into whole names


def test_recommendations_exclude_liked_and_are_sorted(content):
    liked = [content.find_title("The Dark Knight")]
    recs = content.recommend_for(liked, k=10)
    assert len(recs) == 10
    assert liked[0] not in recs.index
    assert list(recs["score"]) == sorted(recs["score"], reverse=True)
    assert recs["score"].between(0, 1).all()


def test_sequels_rank_highly(content):
    recs = content.recommend_for([content.find_title("The Godfather")], k=3)
    assert "The Godfather Part II" in recs["title"].tolist()


def test_no_or_invalid_likes_fall_back_to_popular(content):
    for liked in ([], [-1, 10_000, "x"]):
        recs = content.recommend_for(liked, k=5)
        assert len(recs) == 5
        assert (recs["score"] == 0).all()


def test_find_title_is_fuzzy(content):
    assert content.find_title("the dark knight") == content.find_title("The Dark Knight")
    assert content.find_title("Godfathr") is not None


def test_top_k_handles_exclusions_and_small_catalogs():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k(scores, [1], 2).tolist() == [3, 2]
    assert top_k(scores, [], 10).tolist() == [1, 3, 2, 0]
    assert top_k(scores, [0, 1, 2, 3], 3).tolist() == []
