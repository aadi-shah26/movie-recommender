from pathlib import Path
import re
import ast
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack, csr_matrix

#All the logic for recommending movies

DATA_DIR = Path('data')

def _normalize_col_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())

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
    for candidate in ('actors','cast','starring','castandcrew'):
        if _normalize_col_name(candidate) in cols:
            actor_col = cols[_normalize_col_name(candidate)]
            break

    # year / release column
    year_col = None
    for candidate in ('year','releaseyear','released','year_of_release','release_date'):
        if _normalize_col_name(candidate) in cols:
            year_col = cols[_normalize_col_name(candidate)]
            break

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

    return df.reset_index(drop=True)

class PersonalRecommender:
    def __init__(self, df=None, weights=None):
        self.df = df if df is not None else load_movie_df()

        # feature weights
        w = {'text': 1.0, 'genre': 6.0, 'director': 3.0, 'actors': 4.0, 'year': 2.0}
        if weights:
            w.update(weights)

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

        # director as bag-of-words
        director_series = self.df['director'].astype(str).fillna('').str.strip()
        if director_series.str.len().gt(0).any():
            self.dir_vec = CountVectorizer(lowercase=True)
            director_mat = self.dir_vec.fit_transform(director_series)
        else:
            director_mat = csr_matrix((len(self.df), 0))

        # actors: use custom tokenizer that splits on '|'
        def _actor_tokenizer(s):
            if not s:
                return []
            return [t.strip().lower() for t in s.split('|') if t.strip()]

        actors_series = self.df['actors'].astype(str).fillna('').str.strip()
        if actors_series.str.len().gt(0).any():
            self.actor_vec = CountVectorizer(tokenizer=_actor_tokenizer, lowercase=True)
            actor_mat = self.actor_vec.fit_transform(actors_series)
        else:
            actor_mat = csr_matrix((len(self.df), 0))

        # year numeric scaled into [0,1]
        years = self.df['year'].fillna(self.df['year'].median() if not self.df['year'].isnull().all() else 0)
        ymin, ymax = years.min(), years.max()
        if ymax - ymin == 0:
            year_scaled = np.zeros((len(self.df), 1), dtype=float)
        else:
            year_scaled = ((years - ymin) / (ymax - ymin)).astype(float).values.reshape(-1, 1)
        year_mat = csr_matrix(year_scaled)

        # scale by weights
        text_mat = text_mat.multiply(w['text'])
        genre_mat = genre_mat.multiply(w['genre'])
        director_mat = director_mat.multiply(w['director'])
        actor_mat = actor_mat.multiply(w['actors'])
        year_mat = year_mat.multiply(w['year'])

        # final item matrix (sparse)
        self.matrix = hstack([text_mat, genre_mat, director_mat, actor_mat, year_mat], format='csr')

        # normalized title -> index map
        def _norm(s):
            return re.sub(r'[^a-z0-9]', '', str(s).lower().strip())
        self.title_to_idx = {_norm(t): i for i, t in enumerate(self.df['title'].astype(str))}
        self.liked = []
        self.profile = np.zeros((1, self.matrix.shape[1]), dtype=float)

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

    def recommend(self, k=10):
        if len(self.liked) == 0:
            if 'vote_count' in self.df.columns:
                top = self.df.sort_values('vote_count', ascending=False).head(k)
                return top[['title']].reset_index(drop=True)
            return self.df[['title']].head(k)
        sims = linear_kernel(self.profile, self.matrix).flatten()
        for idx in self.liked:
            sims[idx] = -1
        top_idx = sims.argsort()[::-1][:k]
        return self.df.iloc[top_idx][['title']].assign(score=sims[top_idx]).reset_index(drop=True)

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