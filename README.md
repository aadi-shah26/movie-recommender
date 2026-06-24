# movie-recommender

Backend (FastAPI on port 8000) — from the project root:


cd /Users/aadishah/movie-recommender
source .venv/bin/activate        # first time / if deps changed
uvicorn backend.main:app --reload --port 8000


Frontend (Vite dev server) — in a second terminal:

cd /Users/aadishah/movie-recommender/frontend
npm install                               # first time / if deps changed
npm run dev

cd /Users/aadishah/movie-recommender/frontend
npm run dev