import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    return TestClient(app)


def test_health_reports_content_fallback(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "movies": 1000, "model": "content", "model_version": None}
    assert client.get("/model").json()["model"] == "content"


def test_titles_and_etag(client):
    res = client.get("/titles")
    assert res.status_code == 200
    movies = res.json()
    assert len(movies) == 1000 and movies[0]["title"] == "The Shawshank Redemption"
    assert movies[0]["runtime"] == 142 and movies[0]["rating"] == 9.3
    cached = client.get("/titles", headers={"If-None-Match": res.headers["etag"]})
    assert cached.status_code == 304


def test_recommendations(client):
    res = client.post("/recommendations", json={"liked": [1], "k": 5})
    assert res.status_code == 200
    recs = res.json()
    assert len(recs) == 5 and all(r["id"] != 1 for r in recs)
    assert {"id", "title", "score", "genres", "year", "poster_url"} <= set(recs[0])


@pytest.mark.parametrize("payload", [
    {"liked": [1], "k": 0},
    {"liked": [1], "k": 51},
    {"liked": "nope"},
    {"liked": list(range(201))},
])
def test_recommendations_validates_input(client, payload):
    assert client.post("/recommendations", json=payload).status_code == 422


def test_movie_lookup(client):
    assert client.get("/movie/1").json()["title"] == "The Godfather"
    assert client.get("/movie/1000").status_code == 404
    assert client.get("/movie/-1").status_code == 404
