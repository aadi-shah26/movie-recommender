"""Filesystem locations, resolved from the project root rather than the cwd.

Override with env vars when the data or model artifacts live elsewhere
(e.g. a mounted volume in production).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RECOMMENDER_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("RECOMMENDER_MODELS_DIR", PROJECT_ROOT / "models"))

CATALOG_CSV = DATA_DIR / "IMDb top 1000 movies.csv"
POSTERS_JSON = DATA_DIR / "posters.json"
MOVIELENS_DIR = DATA_DIR / "movielens"
MODEL_ARTIFACT = MODELS_DIR / "model.npz"
MODEL_METADATA = MODELS_DIR / "model.json"
PLOT_EMBEDDINGS = DATA_DIR / "plot_embeddings.npy"
PLOT_EMBEDDINGS_META = DATA_DIR / "plot_embeddings.json"
RANKER_MODEL = MODELS_DIR / "ranker.txt"
