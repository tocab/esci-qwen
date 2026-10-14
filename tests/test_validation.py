import numpy as np
import pytest

from src.common import LABELS
from src.validation import bootstrap_macro_f1, metrics


def test_metrics_and_bootstrap():
    p = np.eye(4)
    report = metrics(LABELS, p)
    assert report["macro_f1"] == 1
    assert report["brier_score"] == 0
    assert report["confusion_matrix"] == np.eye(4, dtype=int).tolist()
    ci = bootstrap_macro_f1(LABELS, p, ["a"] * 4, 20)
    assert ci["lower_95"] == ci["upper_95"] == 1
    assert metrics(["E"], [[1.0, 0, 0, 0]])["macro_f1"] == 0.25
    with pytest.raises(ValueError):
        metrics(LABELS, np.zeros((4, 4)))
