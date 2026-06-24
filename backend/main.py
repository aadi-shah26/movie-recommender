from pathlib import Path
import sys
from typing import List, Dict, Any
import pandas as pd
#helps run frontend and backend together
proj_root = Path(__file__).resolve().parents[1]
src_path = proj_root / 'src'
sys.path.insert(0, str(src_path))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import re

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
# Allow local frontend (different port) to call the API during development
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load data and recommender once
DF = load_movie_df()
RECC = PersonalRecommender(DF)


class RecRequest(BaseModel):
    liked: List[int]
    k: int = 5


@app.get('/titles')
def get_titles(limit: int = 1000):
    # return minimal metadata for client-side search
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
        })
        if len(out) >= limit:
            break
    return out


@app.post('/recommendations')
def recommendations(req: RecRequest):
    # build profile by adding likes incrementally
    # reset recommender profile
    rec = PersonalRecommender(DF)
    for idx in req.liked:
        try:
            rec.add_like(DF.at[int(idx), 'title'])
        except Exception:
            # ignore invalid ids
            continue
    res = rec.recommend(k=req.k)
    out = []
    for _, r in res.iterrows():
        out.append({
            'id': int(r.name) if hasattr(r, 'name') else None,
            'title': r['title'],
            'score': float(r.get('score', 0.0)) if 'score' in r else 0.0,
            'short': DF.at[int(r.name), 'Plot'] if 'Plot' in DF.columns else '',
            'genres': DF.at[int(r.name), 'genres'] if 'genres' in DF.columns else DF.at[int(r.name), 'Genre'] if 'Genre' in DF.columns else '',
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
