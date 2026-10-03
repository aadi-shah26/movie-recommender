import os
import tempfile

import pytest

# Point the API at an empty models dir so tests exercise the documented
# content-only fallback and never depend on a locally trained artifact.
os.environ["RECOMMENDER_MODELS_DIR"] = tempfile.mkdtemp(prefix="recommender-test-models-")

from recommender.catalog import load_movie_df  # noqa: E402
from recommender.content import ContentRecommender  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    return load_movie_df()


@pytest.fixture(scope="session")
def content(catalog):
    return ContentRecommender(catalog)
