"""CPU classifiers and explicit, training-only cross-validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline


def build_baseline_classifier(model_name: str, random_state: int = 42) -> Pipeline:
    """Build the fixed Stage 4A estimators, independently of tuning defaults."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    if model_name == "svm":
        from src.baseline_scaling import BatchedStandardScaler

        return Pipeline([
            ("scaler", BatchedStandardScaler(copy=False)),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                               class_weight=None, random_state=random_state)),
        ])
    if model_name == "random_forest":
        return Pipeline([("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, class_weight=None, random_state=random_state,
            n_jobs=1,
        ))])
    raise ValueError("Model must be 'svm' or 'random_forest'.")


def build_classifier(model_name: str, random_state: int = 42) -> Pipeline:
    """Keep SVM scaling in the pipeline to avoid cross-validation leakage."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if model_name == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("classifier", SVC(class_weight="balanced"))])
    if model_name == "random_forest":
        return Pipeline([("classifier", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_state,
            n_jobs=1))])
    raise ValueError("Model must be 'svm' or 'random_forest'.")


def optimize_classifier(x_train, y_train, model_name: str, parameter_grid=None,
                        folds: int = 5, n_jobs: int = 1,
                        random_state: int = 42) -> GridSearchCV:
    """Fit only when explicitly called; never pass held-out test data here."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    defaults = {
        "svm": {"classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"]},
        "random_forest": {"classifier__max_depth": [None, 10, 20],
                          "classifier__min_samples_leaf": [1, 2]},
    }
    estimator = build_classifier(model_name, random_state)
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    search = GridSearchCV(
        estimator,
        defaults[model_name] if parameter_grid is None else parameter_grid,
        scoring="f1", cv=cv, n_jobs=n_jobs,
        error_score="raise", return_train_score=False,
    )
    return search.fit(x_train, y_train)


def save_model(model, path: Path, *, metadata: dict) -> None:
    """Save estimator and preprocessing/feature configuration together."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_model(path: Path) -> dict:
    """Load only trusted Joblib artifacts; deserialization can execute code."""
    import joblib

    return joblib.load(Path(path))
