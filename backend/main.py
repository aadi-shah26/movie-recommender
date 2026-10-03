"""Movie Recommender API.

Serves the trained two-stage model (models/, from `python -m scripts.train`):
EASE + content candidates reranked by LightGBM. If no trained model is present
it falls back to the content-only recommender, so the API still works on a
fresh clone.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
"""
import hashlib
import json
import logging
import os
import re
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from recommender.catalog import load_movie_df
from recommender.content import ContentRecommender
from recommender.model import TrainedModel
from recommender.paths import MODELS_DIR, POSTERS_JSON

log = logging.getLogger("uvicorn.error")

MAX_K = 50
MAX_LIKED = 200


# ---- schemas ---------------------------------------------------------------

class Movie(BaseModel):
    id: int
    title: str
    year: int | None = None
    short: str | None = None
    genres: str | None = None
    rating: float | None = None
    certificate: str | None = None
    runtime: int | None = None
    votes: int | None = None
    director: str | None = None
    poster_url: str | None = None


class Recommendation(Movie):
    score: float


class RecRequest(BaseModel):
    liked: Annotated[list[int], Field(max_length=MAX_LIKED)]
    k: Annotated[int, Field(ge=1, le=MAX_K)] = 5


class ModelInfo(BaseModel):
    model: str
    version: str | None = None
    params: dict = {}
    trained_at: str | None = None
    dataset: str | None = None
    test_metrics: dict | None = None
    cold_start_metrics: dict | None = None
    feature_importance: dict | None = None


class Health(BaseModel):
    status: str
    movies: int
    model: str
    model_version: str | None = None


# ---- model + catalog, loaded once at startup --------------------------------

DF = load_movie_df()

try:
    MODEL = TrainedModel.load(DF)
    MODEL_INFO = ModelInfo(
        model=MODEL.name,
        **{k: MODEL.metadata.get(k) for k in (
            "version", "params", "trained_at", "dataset", "test_metrics", "cold_start_metrics", "feature_importance")},
    )
except (FileNotFoundError, KeyError, ValueError) as e:
    log.warning("No usable trained model in %s (%s); serving content-only. "
                "Run `python -m scripts.train` to train one.", MODELS_DIR, e)
    MODEL = ContentRecommender(DF)
    MODEL_INFO = ModelInfo(model="content", params={"quality": MODEL.quality_weight})

# Optional poster map produced by scripts/fetch_posters.py ({title: poster_url}).
try:
    POSTERS = json.loads(POSTERS_JSON.read_text()) if POSTERS_JSON.exists() else {}
except (OSError, json.JSONDecodeError):
    POSTERS = {}


def _num(x, cast=float):
    if x is None or pd.isna(x):
        return None
    try:
        return cast(float(x))
    except (TypeError, ValueError):
        return None


def _str(x):
    if x is None or pd.isna(x):
        return None
    return str(x).strip() or None


def _runtime(x):
    m = re.search(r"(\d+)", str(x)) if _str(x) else None
    return int(m.group(1)) if m else None


def _year(i, row):
    year = _num(row["year"], int)
    if year is None and getattr(MODEL, "years", None) is not None and MODEL.years[i] > 0:
        year = int(MODEL.years[i])  # catalog has no year column; MovieLens does
    return year


def _movie(i: int) -> dict:
    row = DF.iloc[i]
    return {
        "id": i,
        "title": row["title"],
        "year": _year(i, row),
        "short": _str(row.get("Plot")),
        "genres": _str(row["genres"]),
        "rating": _num(row["rating"]),
        "certificate": _str(row.get("Certificate")),
        "runtime": _runtime(row.get("Duration")),
        "votes": _num(row["votes"], int),
        "director": _str(row["director"]),
        "poster_url": POSTERS.get(row["title"]),
    }


# The catalog is static for the life of the process: build the list once and
# give it a stable ETag so browsers can revalidate instead of re-downloading.
TITLES = [_movie(i) for i in range(len(DF))]
TITLES_ETAG = '"' + hashlib.md5(json.dumps(TITLES, sort_keys=True).encode()).hexdigest() + '"'
TITLES_CACHE_CONTROL = "public, max-age=3600"


# ---- app -------------------------------------------------------------------

app = FastAPI(title="Movie Recommender API", version=MODEL_INFO.version or "content")

# Comma-separated CORS_ORIGINS lets production lock this down without code
# changes. "*" is not allowed together with allow_credentials=True.
_DEFAULT_ORIGINS = (
    "http://localhost:8080,http://localhost:8081,http://127.0.0.1:8080,http://127.0.0.1:8081,"
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=Health)
def health():
    """Liveness/readiness probe for load balancers and monitoring."""
    return Health(status="ok", movies=len(DF), model=MODEL_INFO.model, model_version=MODEL_INFO.version)


@app.get("/model", response_model=ModelInfo)
def model_info():
    """Which model is serving, its hyperparameters and offline test metrics."""
    return MODEL_INFO


@app.get("/titles", response_model=list[Movie])
def get_titles(request: Request):
    headers = {"ETag": TITLES_ETAG, "Cache-Control": TITLES_CACHE_CONTROL}
    if request.headers.get("if-none-match") == TITLES_ETAG:
        return Response(status_code=304, headers=headers)
    return JSONResponse(content=TITLES, headers=headers)


@app.post("/recommendations", response_model=list[Recommendation])
def recommendations(req: RecRequest):
    res = MODEL.recommend_for(req.liked, k=req.k)
    return [{**TITLES[int(i)], "score": float(score)} for i, score in zip(res.index, res["score"], strict=True)]


@app.get("/movie/{movie_id}", response_model=Movie)
def get_movie(movie_id: int):
    if not 0 <= movie_id < len(DF):
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return TITLES[movie_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
