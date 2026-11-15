"""Binary classification metrics; defective images are the positive class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def evaluate_predictions(
    y_true, y_pred, y_score=None,
) -> tuple[dict, np.ndarray]:
    """Accept positive-class probabilities or decision function scores."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    truth, predictions = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or predictions.shape != truth.shape or truth.size == 0:
        raise ValueError("Expected equally sized, nonempty label vectors.")
    observed_labels = set(np.unique(np.concatenate((truth, predictions))))
    if not observed_labels.issubset({0, 1}):
        raise ValueError("Labels must be 0 (acceptable) or 1 (defective).")
    metrics = {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision_score(truth, predictions, zero_division=0),
        "recall": recall_score(truth, predictions, zero_division=0),
        "f1": f1_score(truth, predictions, zero_division=0),
        "roc_auc": float("nan"),
    }
    if y_score is not None:
        scores = np.asarray(y_score)
        if scores.shape != truth.shape or not np.isfinite(scores).all():
            raise ValueError(
                "Scores must be a finite vector matching the labels."
            )
        if np.unique(truth).size == 2:
            metrics["roc_auc"] = roc_auc_score(truth, scores)
    return metrics, confusion_matrix(truth, predictions, labels=[0, 1])


def save_comparison(results: dict[str, dict], path: Path) -> pd.DataFrame:
    """Save actual model metrics supplied by the caller, ordered by F1."""
    import pandas as pd

    table = pd.DataFrame.from_dict(results, orient="index").rename_axis(
        "model"
    )
    table = table.sort_values("f1", ascending=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return table
