import numpy as np
from scipy.sparse import csr_matrix

from recommender.collaborative import EASE

# Items 0 and 1 are always liked together; item 2 goes with 3; item 4 unseen.
X = csr_matrix(np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
], dtype=np.float32))


def test_ease_learns_co_occurrence():
    model = EASE(l2=1.0).fit(X)
    assert np.allclose(np.diag(model.B), 0)
    scores = model.scores([0])
    assert scores[1] > scores[3]
    assert model.scores([2]).argmax() == 3


def test_ease_coverage_and_empty_query():
    model = EASE(l2=1.0).fit(X)
    assert model.covered.tolist() == [True, True, True, True, False]
    assert np.all(model.scores([]) == 0)
