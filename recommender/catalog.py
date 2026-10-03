"""Load the IMDb top-1000 catalog into a normalized DataFrame.

Movie ids used throughout the app (API, frontend favorites, model matrices)
are the row positions of this DataFrame, so row order must stay stable.
"""
import ast
import re

import numpy as np
import pandas as pd

from .paths import CATALOG_CSV


def normalize_key(s) -> str:
    """Lowercase alphanumeric-only key, used for fuzzy title/column matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower().strip())


def _find_col(cols, candidates):
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in cols:
            return cols[key]
    return None


def _normalize_actors_cell(x):
    """Turn "['A', 'B']" (or "A, B") into the pipe-joined "A | B"."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return " | ".join(str(v).strip() for v in val if str(v).strip())
    except (ValueError, SyntaxError):
        pass
    s2 = re.sub(r"[\[\]']", "", s)
    parts = [p.strip() for p in s2.split(",") if p.strip()]
    return " | ".join(parts) if parts else s


def _extract_year(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(19\d{2}|20\d{2})", str(x))
    return int(m.group(0)) if m else np.nan


def load_movie_df(csv_path=CATALOG_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {normalize_key(c): c for c in df.columns}

    title_col = _find_col(cols, ("title", "moviename", "movie", "name", "movie_title"))
    if title_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == object]
        title_col = object_cols[0] if object_cols else df.columns[0]

    text_col = _find_col(cols, ("plot", "overview", "description", "summary", "story"))
    genre_col = _find_col(cols, ("genres", "genre", "categories"))
    director_col = _find_col(cols, ("director", "directors", "director_name"))
    actor_col = _find_col(cols, ("actors", "cast", "starring", "castandcrew", "stars"))
    year_col = _find_col(cols, ("year", "releaseyear", "released", "year_of_release", "release_date"))
    rating_col = _find_col(cols, ("imdbrating", "rating", "imdb_rating", "averagerating"))
    votes_col = _find_col(cols, ("votes", "numvotes", "votecount", "vote_count"))
    metascore_col = _find_col(cols, ("metascore", "metacritic"))

    df = df.rename(columns={title_col: "title"})
    df["title"] = df["title"].astype(str).str.strip()

    text = df["title"]
    if text_col:
        text = text + " " + df[text_col].fillna("").astype(str)
    df["text"] = text.str.strip()

    df["genres"] = df[genre_col].fillna("").astype(str) if genre_col else ""
    df["director"] = df[director_col].fillna("").astype(str) if director_col else ""
    df["actors"] = df[actor_col].apply(_normalize_actors_cell) if actor_col else ""
    df["year"] = df[year_col].apply(_extract_year) if year_col else np.nan

    # Numeric quality signals; strip thousands separators etc. before coercing.
    def _to_numeric(col):
        if not col:
            return np.nan
        cleaned = df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    df["rating"] = _to_numeric(rating_col)
    df["votes"] = _to_numeric(votes_col)
    df["metascore"] = _to_numeric(metascore_col)

    return df.reset_index(drop=True)
