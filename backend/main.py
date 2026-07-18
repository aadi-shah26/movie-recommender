from pathlib import Path
import os
import sys
from typing import List, Dict, Any, Annotated
import pandas as pd
#helps run frontend and backend together
proj_root = Path(__file__).resolve().parents[1]
src_path = proj_root / 'src'
sys.path.insert(0, str(src_path))

import uvicorn
from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import re
import json
import hashlib

from personal_recommender import load_movie_df, PersonalRecommender


def _to_float(x):
    """Safely coerce a cell to float, returning None for missing/invalid."""
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x):
    f = _to_float(x)
    return int(f) if f is not None else None


def _parse_runtime(x):
    """Extract minutes from values like '142 min'."""
    if x is None or pd.isna(x):
        return None
    m = re.search(r'(\d+)', str(x))
    return int(m.group(1)) if m else None


def _clean_str(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    return s or None

app = FastAPI(title='Movie Recommender API')

# Allowed CORS origins are driven by the CORS_ORIGINS env var (comma-separated)
# so production can lock this down without code changes. Defaults cover the
# local dev frontend ports. Note: a wildcard "*" is intentionally NOT included
# because it is invalid together with allow_credentials=True.
_DEFAULT_ORIGINS = (
    "http://localhost:8080,http://localhost:8081,"
    "http://127.0.0.1:8080,http://127.0.0.1:8081,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get('CORS_ORIGINS', _DEFAULT_ORIGINS).split(',')
    if o.strip()
]

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load data and recommender once
DF = load_movie_df()
RECC = PersonalRecommender(DF)

# Optional poster map produced by src/fetch_posters.py ({title: poster_url}).
# Absent until the one-time backfill is run; the frontend falls back to its
# generated tile for any title without a poster.
_POSTERS_PATH = proj_root / 'data' / 'posters.json'
try:
    POSTERS = json.loads(_POSTERS_PATH.read_text()) if _POSTERS_PATH.exists() else {}
except Exception:
    POSTERS = {}


@app.get('/health')
def health():
    """Lightweight liveness/readiness probe for load balancers and monitoring."""
    return {
        'status': 'ok',
        'movies': int(len(DF)),
        'model_ready': RECC is not None and getattr(RECC, 'matrix', None) is not None,
    }


class RecRequest(BaseModel):
    liked: List[int]
    k: int = 5


def _build_titles():
    """Build the full minimal-metadata list for client-side search.

    The catalog is static for the lifetime of the process, so we compute this
    once at startup and serve the cached copy instead of re-iterating the
    dataframe on every request.
    """
    out = []
    for i, row in DF.iterrows():
        out.append({
            'id': int(i),
            'title': row['title'],
            'year': int(row['year']) if 'year' in row and not pd.isna(row['year']) else None,
            'short': row.get('Plot') if 'Plot' in row else row.get('short', ''),
            'genres': row.get('genres') if 'genres' in row else row.get('Genre', ''),
            'rating': _to_float(row.get('IMDb Rating')),
            'certificate': _clean_str(row.get('Certificate')),
            'runtime': _parse_runtime(row.get('Duration')),
            'votes': _to_int(row.get('Votes')),
            'director': _clean_str(row.get('director') if 'director' in row else row.get('Director')),
            'poster_url': POSTERS.get(row['title']),
        })
    return out


# Precompute once at startup + a stable ETag for conditional requests.
_TITLES_CACHE = _build_titles()
_TITLES_ETAG = '"' + hashlib.md5(
    json.dumps(_TITLES_CACHE, sort_keys=True, default=str).encode()
).hexdigest() + '"'
_TITLES_CACHE_CONTROL = 'public, max-age=3600'


@app.get('/titles')
def get_titles(
    request: Request,
    limit: Annotated[int | None, Query(ge=1)] = None,
):
    effective_limit = len(_TITLES_CACHE) if limit is None else min(limit, len(_TITLES_CACHE))
    full = effective_limit >= len(_TITLES_CACHE)

    # For full-list responses, let the browser skip the download entirely
    # when its cached copy is still current.
    if full and request.headers.get('if-none-match') == _TITLES_ETAG:
        return Response(
            status_code=304,
            headers={'ETag': _TITLES_ETAG, 'Cache-Control': _TITLES_CACHE_CONTROL},
        )

    headers = {'Cache-Control': _TITLES_CACHE_CONTROL}
    if full:
        headers['ETag'] = _TITLES_ETAG
    return JSONResponse(content=_TITLES_CACHE[:effective_limit], headers=headers)


@app.post('/recommendations')
def recommendations(req: RecRequest):
    # Reuse the pre-built matrix (RECC) and compute the profile on the fly.
    # No per-request model rebuild and no shared-state mutation, so this is
    # cheap and safe under concurrent requests.
    liked_ids = []
    for idx in req.liked:
        try:
            liked_ids.append(int(idx))
        except (TypeError, ValueError):
            continue
    res = RECC.recommend_for(liked_ids, k=req.k)
    out = []
    for _, r in res.iterrows():
        out.append({
            'id': int(r.name) if hasattr(r, 'name') else None,
            'title': r['title'],
            'score': float(r.get('score', 0.0)) if 'score' in r else 0.0,
            'short': DF.at[int(r.name), 'Plot'] if 'Plot' in DF.columns else '',
            'genres': DF.at[int(r.name), 'genres'] if 'genres' in DF.columns else DF.at[int(r.name), 'Genre'] if 'Genre' in DF.columns else '',
            'poster_url': POSTERS.get(r['title']),
        })
    return out


@app.get('/movie/{movie_id}')
def get_movie(movie_id: int):
    if movie_id < 0 or movie_id >= len(DF):
        return {}
    row = DF.iloc[movie_id]
    return {
        'id': movie_id,
        'title': row['title'],
        'year': int(row['year']) if 'year' in row and not pd.isna(row['year']) else None,
        'genres': row.get('genres') if 'genres' in row else row.get('Genre', ''),
        'short': row.get('Plot') or '',
    }


if __name__ == '__main__':
    import pandas as pd
    uvicorn.run('backend.main:app', host='127.0.0.1', port=8000, reload=False)
