# Backend API image. The model is trained at build time (downloads MovieLens,
# ~1 MB), so the image ships with a ready-to-serve artifact.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY recommender/ recommender/
COPY scripts/ scripts/
COPY backend/ backend/
COPY ["data/IMDb top 1000 movies.csv", "data/posters.json", "data/"]

RUN python -m scripts.train && rm -rf data/movielens

RUN useradd --create-home app && chown -R app /app
USER app

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
