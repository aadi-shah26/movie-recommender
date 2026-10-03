import pandas as pd

from recommender import movielens

ML_MOVIES = pd.DataFrame({
    "movieId": [1, 2, 3, 4, 5],
    "title": [
        "Shawshank Redemption, The (1994)",
        "Lives of Others, The (Leben der Anderen, Das) (2006)",
        "Lock, Stock & Two Smoking Barrels (1998)",
        "Hamlet (1948)",
        "Hamlet (1996)",
    ],
})
ML_RATINGS = pd.DataFrame({
    "userId": [1, 1, 2, 2, 2, 3],
    "movieId": [1, 5, 1, 2, 5, 4],
    "rating": [5.0, 4.0, 3.5, 4.5, 4.0, 5.0],
})


def test_title_key_normalizes_articles_accents_and_ampersands():
    assert movielens.title_key("The Shawshank Redemption") == movielens.title_key("Shawshank Redemption")
    assert movielens.title_key("Amélie") == "amelie"
    assert movielens.title_key("Lock, Stock and Two Smoking Barrels") == movielens.title_key(
        "Lock, Stock & Two Smoking Barrels")


def test_match_catalog():
    catalog = ["The Shawshank Redemption", "The Lives of Others", "Lock, Stock and Two Smoking Barrels",
               "Hamlet", "Some Unknown Film"]
    ids, years = movielens.match_catalog(catalog, ML_MOVIES, ML_RATINGS)
    assert ids.tolist() == [1, 2, 3, 5, -1]  # ambiguous "Hamlet" -> the more-rated one
    assert years.tolist() == [1994, 2006, 1998, 1996, -1]


def test_interactions_keep_only_positive_catalog_ratings():
    X, users = movielens.interactions(ML_RATINGS, [1, 2, 5])
    assert users.tolist() == [1, 2]  # user 3 only rated a non-catalog movie
    assert X.toarray().tolist() == [[1, 0, 1], [0, 1, 1]]  # 3.5 is below the like threshold
