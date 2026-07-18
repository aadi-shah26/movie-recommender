from pathlib import Path
import re
import ast
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

#All the logic for recommending movies

DATA_DIR = Path('data')

def _normalize_col_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())


def _name_tokenizer(s):
    """Tokenize a names field into whole-name tokens.

    Splits on '|' and ',' so multi-name fields break apart, but a single
    name like "Christopher Nolan" stays one token (lowercased). Used for both
    the director and actor vectorizers so two different people who merely share
    a first/last name don't spuriously match.
    """
    if not s:
        return []
    parts = re.split(r'[|,]', str(s))
    return [p.strip().lower() for p in parts if p.strip()]

def load_movie_df():
    csv = next(DATA_DIR.rglob('*.csv'))
    df = pd.read_csv(csv)

    # map original column names by normalized key
    cols = { _normalize_col_name(c): c for c in df.columns }

    # title column
    for key in ('title','moviename','movie','name','movie_title'):
        if key in cols:
            title_col = cols[key]
            break
    else:
        title_col = None
        for c in df.columns:
            if df[c].dtype == object:
                title_col = c
                break
        if title_col is None:
            title_col = df.columns[0]

    # possible text fields
    text_col = None
    for candidate in ('plot','overview','description','summary','story'):
        if _normalize_col_name(candidate) in cols:
            text_col = cols[_normalize_col_name(candidate)]
            break

    # genre column
    genre_col = None
    for candidate in ('genres','genre','categories'):
        if _normalize_col_name(candidate) in cols:
            genre_col = cols[_normalize_col_name(candidate)]
            break

    # director column
    director_col = None
    for candidate in ('director','directors','director_name'):
        if _normalize_col_name(candidate) in cols:
            director_col = cols[_normalize_col_name(candidate)]
            break

    # actors column
    actor_col = None
    for candidate in ('actors','cast','starring','castandcrew','stars'):
        if _normalize_col_name(candidate) in cols:
            actor_col = cols[_normalize_col_name(candidate)]
            break

    # year / release column
    year_col = None
    for candidate in ('year','releaseyear','released','year_of_release','release_date'):
        if _normalize_col_name(candidate) in cols:
            year_col = cols[_normalize_col_name(candidate)]
            break

    # numeric quality columns (rating / votes / metascore)
    def _find_col(candidates):
        for candidate in candidates:
            if _normalize_col_name(candidate) in cols:
                return cols[_normalize_col_name(candidate)]
        return None

    rating_col = _find_col(('imdbrating', 'rating', 'imdb_rating', 'averagerating'))
    votes_col = _find_col(('votes', 'numvotes', 'votecount', 'vote_count'))
    metascore_col = _find_col(('metascore', 'metacritic'))

    # rename title to 'title'
    df = df.rename(columns={title_col: 'title'})

    # build simple text field (title + text_col)
    text = df['title'].astype(str).fillna('').str.strip()
    if text_col:
        text = text + ' ' + df[text_col].astype(str).fillna('')
    df['text'] = text.str.strip()
    df['title'] = df['title'].astype(str).str.strip()

    # genres
    df['genres'] = df[genre_col].astype(str).fillna('') if genre_col else ''

    # director
    df['director'] = df[director_col].astype(str).fillna('') if director_col else ''

    # actors: try to parse list-like values, fallback to simple cleanup
    def _normalize_actors_cell(x):
        if pd.isna(x):
            return ''
        s = str(x).strip()
        # try literal_eval for strings like "['A','B']"
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                # join with pipe delimiter to preserve whole names as tokens
                return ' | '.join([str(v).strip() for v in val if str(v).strip()])
        except Exception:
            pass
        # fallback: remove brackets/quotes then split on commas
        s2 = re.sub(r'[\[\]\']', '', s)
        parts = [p.strip() for p in s2.split(',') if p.strip()]
        if parts:
            return ' | '.join(parts)
        # if no commas, return original
        return s

    df['actors'] = df[actor_col].apply(_normalize_actors_cell) if actor_col else ''

    # year extraction
    def _extract_year(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        m = re.search(r'(19\d{2}|20\d{2})', s)
        return int(m.group(0)) if m else np.nan
    df['year'] = df[year_col].apply(_extract_year) if year_col else np.nan

    # numeric quality signals, coerced to numbers (strip stray non-numeric chars
    # like thousands separators from vote counts)
    def _to_numeric(col):
        if not col:
            return np.nan
        cleaned = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')

    df['rating'] = _to_numeric(rating_col)
    df['votes'] = _to_numeric(votes_col)
    df['metascore'] = _to_numeric(metascore_col)

    return df.reset_index(drop=True)

class PersonalRecommender:
    def __init__(self, df=None, weights=None, normalize_features=True):
        self.df = df if df is not None else load_movie_df()
        self.normalize_features = normalize_features

        # feature weights. 'quality' is the blend weight for the popularity/
        # rating prior applied at ranking time (not a feature block).
        w = {'text': 1.0, 'genre': 6.0, 'director': 3.0, 'actors': 4.0,
             'year': 2.0, 'quality': 0.15}
        if weights:
            w.update(weights)
        self.quality_weight = float(w['quality'])

        # text TF-IDF
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
        text_mat = self.tfidf.fit_transform(self.df['text'].fillna(''))

        # genres as bag-of-tokens (split on comma/pipe)
        genres_series = self.df['genres'].astype(str).fillna('').str.strip()
        if genres_series.str.len().gt(0).any():
            self.genre_vec = CountVectorizer(token_pattern=r"[^,|\s]+")
            genre_mat = self.genre_vec.fit_transform(genres_series)
        else:
            genre_mat = csr_matrix((len(self.df), 0))

        # director: whole-name tokens (so "Christopher Nolan" is one token, not
        # two words that collide with other directors named Christopher).
        director_series = self.df['director'].astype(str).fillna('').str.strip()
        if director_series.str.len().gt(0).any():
            self.dir_vec = CountVectorizer(tokenizer=_name_tokenizer, token_pattern=None, lowercase=False)
            director_mat = self.dir_vec.fit_transform(director_series)
        else:
            director_mat = csr_matrix((len(self.df), 0))

        # actors: whole-name tokens (already pipe-joined in load_movie_df)
        actors_series = self.df['actors'].astype(str).fillna('').str.strip()
        if actors_series.str.len().gt(0).any():
            self.actor_vec = CountVectorizer(tokenizer=_name_tokenizer, token_pattern=None, lowercase=False)
            actor_mat = self.actor_vec.fit_transform(actors_series)
        else:
            actor_mat = csr_matrix((len(self.df), 0))

        # year numeric scaled into [0,1] -- only when a usable year exists,
        # otherwise emit a 0-column block so the dead feature contributes nothing
        if self.df['year'].notna().any():
            years = self.df['year'].fillna(self.df['year'].median())
            ymin, ymax = years.min(), years.max()
        else:
            ymin = ymax = 0
        if ymax - ymin == 0:
            year_mat = csr_matrix((len(self.df), 0))
        else:
            year_scaled = ((years - ymin) / (ymax - ymin)).astype(float).values.reshape(-1, 1)
            year_mat = csr_matrix(year_scaled)

        # per-block L2 normalization (so weights mean relative importance rather
        # than being swamped by token counts / vector length), then weight.
        blocks = [
            (text_mat, w['text']),
            (genre_mat, w['genre']),
            (director_mat, w['director']),
            (actor_mat, w['actors']),
            (year_mat, w['year']),
        ]
        scaled = []
        for block, weight in blocks:
            if block.shape[1] == 0:
                scaled.append(block)
                continue
            if self.normalize_features:
                block = normalize(block, norm='l2', axis=1)
            scaled.append(block.multiply(weight))

        # final item matrix (sparse); L2-normalize rows so linear_kernel == cosine
        self.matrix = hstack(scaled, format='csr')
        if self.normalize_features:
            self.matrix = normalize(self.matrix, norm='l2', axis=1)

        # quality/popularity prior in [0,1] from rating + log(votes)
        self.quality = self._build_quality()

        # normalized title -> index map
        def _norm(s):
            return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())
        self.title_to_idx = {_norm(t): i for i, t in enumerate(self.df['title'].astype(str))}
        self.liked = []
        self.profile = np.zeros((1, self.matrix.shape[1]), dtype=float)

    def _build_quality(self):
        """Combine normalized IMDb rating and log-votes into a [0,1] prior."""
        n = len(self.df)

        def _minmax(series):
            s = pd.to_numeric(series, errors='coerce')
            if s.notna().sum() == 0:
                return np.zeros(n)
            s = s.fillna(s.median())
            lo, hi = s.min(), s.max()
            if hi - lo == 0:
                return np.zeros(n)
            return ((s - lo) / (hi - lo)).to_numpy()

        rating_q = _minmax(self.df['rating']) if 'rating' in self.df.columns else np.zeros(n)
        if 'votes' in self.df.columns:
            votes_q = _minmax(np.log1p(pd.to_numeric(self.df['votes'], errors='coerce')))
        else:
            votes_q = np.zeros(n)
        return 0.5 * rating_q + 0.5 * votes_q

    def _find_title(self, title):
        key = re.sub(r'[^a-z0-9]', '', str(title).lower().strip())
        if key in self.title_to_idx:
            return self.title_to_idx[key]
        choices = list(self.title_to_idx.keys())
        matches = difflib.get_close_matches(key, choices, n=3, cutoff=0.5)
        if matches:
            return self.title_to_idx[matches[0]]
        return None

    def add_like(self, title):
        idx = self._find_title(title)
        if idx is None:
            raise KeyError(f"Title not found (no close match): {title}")
        if idx in self.liked:
            return idx, False
        vec = self.matrix[idx].toarray()
        n = len(self.liked)
        if n == 0:
            self.profile = vec.astype(float)
        else:
            self.profile = (self.profile * n + vec) / (n + 1)
        self.liked.append(idx)
        return idx, True

    def _popular(self, k):
        """Fallback when there are no likes: most popular / best-rated."""
        for col in ('votes', 'rating'):
            if col in self.df.columns and self.df[col].notna().any():
                top = self.df.sort_values(col, ascending=False).head(k)
                return top[['title']].assign(score=0.0)
        return self.df[['title']].head(k).assign(score=0.0)

    def _rank(self, profile, exclude_ids, k):
        """Score every movie against ``profile`` and return the top-k.

        When features are normalized, the content score is true cosine
        similarity in [0,1]; a convex blend with the quality prior nudges
        ranking toward better-regarded films while keeping the result in [0,1].
        Keeps the original DataFrame index so callers can map rows back to
        their real movie ids (do NOT reset_index here).
        """
        profile = np.asarray(profile, dtype=float).reshape(1, -1)
        if self.normalize_features:
            profile = normalize(profile, norm='l2', axis=1)
        content = linear_kernel(profile, self.matrix).flatten()

        beta = self.quality_weight
        final = (1 - beta) * content + beta * self.quality if beta else content

        for i in exclude_ids:
            if 0 <= i < len(final):
                final[i] = -1.0
        top_idx = final.argsort()[::-1][:k]
        return self.df.iloc[top_idx][['title']].assign(score=final[top_idx])

    def recommend(self, k=10):
        if len(self.liked) == 0:
            return self._popular(k)
        return self._rank(self.profile, self.liked, k)

    def recommend_for(self, liked_ids, k=10):
        """Stateless recommendation against the pre-built matrix.

        Builds the taste profile on the fly from ``liked_ids`` (movie ids,
        which are row positions in the matrix) without mutating any instance
        state, so a single shared recommender can serve concurrent requests.
        """
        n_items = self.matrix.shape[0]
        # validate + dedupe while preserving order
        liked = list(dict.fromkeys(
            i for i in liked_ids
            if isinstance(i, (int, np.integer)) and 0 <= i < n_items
        ))

        if not liked:
            return self._popular(k)

        # profile = mean of the liked items' feature vectors
        profile = np.asarray(self.matrix[liked].mean(axis=0))
        return self._rank(profile, liked, k)

if __name__ == '__main__':
    df = load_movie_df()
    print(f'Loaded {len(df)} movies.')
    rec = PersonalRecommender(df)
    print('Enter up to 5 favorite movie titles, one per line. Type "quit" to exit early.\n')
    count = 0
    while count < 5:
        title = input(f'Favorite #{count+1}: ').strip()
        if not title or title.lower() in ('quit', 'exit'):
            break
        try:
            idx, added = rec.add_like(title)
        except KeyError:
            key = re.sub(r'[^a-z0-9]', '', title.lower().strip())
            choices = list(rec.title_to_idx.keys())
            matches = difflib.get_close_matches(key, choices, n=5, cutoff=0.4)
            if matches:
                print('No exact match. Did you mean one of these?')
                for i, m in enumerate(matches, 1):
                    print(f'  {i}. {rec.df.iloc[rec.title_to_idx[m]]["title"]}')
                sel = input('Choice (enter to retry): ').strip()
                if sel.isdigit() and 1 <= int(sel) <= len(matches):
                    chosen = rec.df.iloc[rec.title_to_idx[matches[int(sel)-1]]]["title"]
                    idx, added = rec.add_like(chosen)
                    if added:
                        count += 1
                else:
                    print('Please try typing the title again.')
                    continue
            else:
                print('Title not found. Please try again.')
                continue
        else:
            if not added:
                print('Already added that movie; choose another.')
                continue
            count += 1

        print('\nTop recommendations now:')
        print(rec.recommend(k=10).to_string(index=False))
        print('---\n')

    print('Done. Final recommendations:')
    print(rec.recommend(k=20).to_string(index=False))