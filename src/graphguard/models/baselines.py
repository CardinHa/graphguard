"""
Non-GNN baseline classifiers for comparison.

Both baselines use the same handcrafted node features as the GNN but do NOT
perform graph message passing.  This isolates the value of relational
(neighborhood) information over purely local structural features.

Models
------
  LogisticRegression  — linear decision boundary, fast, interpretable
  RandomForest        — non-linear, handles feature interactions, robust
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from graphguard.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BaselinePredictions:
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


class BaselineModels:
    """
    Trains logistic regression and random forest on the node feature matrix.

    Usage
    -----
    >>> bl = BaselineModels()
    >>> results = bl.fit_predict(X_train, y_train, X_test, y_test)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._models: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        }

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> list[BaselinePredictions]:
        """
        Fit all baselines on training data and predict on test data.

        Returns a list of BaselinePredictions, one per model.
        """
        # Logistic regression benefits from feature scaling
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        results = []
        for name, model in self._models.items():
            logger.info(f"Training {name}...")

            if name == "LogisticRegression":
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            results.append(BaselinePredictions(
                model_name=name,
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
            ))
            logger.info(f"{name} training complete.")

        return results

    def feature_importances(
        self, feature_names: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Return RandomForest feature importances, sorted descending.

        If feature_names is provided, the returned frame is indexed by the
        named features so the report is human-readable.
        """
        rf = self._models.get("RandomForest")
        if rf is None or not hasattr(rf, "feature_importances_"):
            return None
        imps = rf.feature_importances_
        names = feature_names if feature_names is not None else list(range(len(imps)))
        return (
            pd.DataFrame({"feature": names, "importance": imps})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
