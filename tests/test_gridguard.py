import numpy as np

from gridguard.conformal import classwise_thresholds, decide, sets


def test_classwise_coverage_on_synthetic_probs():
    rng = np.random.default_rng(0)

    def draw(n):
        y = rng.integers(0, 2, n)
        p1 = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.25, n), 0.01, 0.99)
        return np.c_[1 - p1, p1], y

    Pc, yc = draw(6000)
    Pt, yt = draw(8000)
    S = sets(Pt, classwise_thresholds(Pc, yc, 0.1))
    for c in (0, 1):
        assert abs(S[yt == c, c].mean() - 0.9) < 0.03


def test_decide_refers_empty_and_double():
    S = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    assert list(decide(S)) == [0, 1, -1, -1]
