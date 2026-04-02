"""
detectors/ml_detector.py — ML-based detectors: Isolation Forest + Gradient Boosting.

Layer 2: catches sophisticated bots that evade statistical rules.

Isolation Forest (Unsupervised)
────────────────────────────────
Anomaly detection without labels. Builds random trees and measures
how quickly a session can be isolated. Anomalies (bots) are easier
to isolate → shorter average path length → higher anomaly score.

Advantage: works on day-0 with no labelled data.
Disadvantage: can't distinguish bot types; tuning contamination is tricky.

Gradient Boosting (Supervised)
────────────────────────────────
Trained on labelled sessions (human vs bot). Learns complex non-linear
decision boundaries. Feature importances reveal *why* a session is flagged.

Advantage: highest accuracy when labelled data is available.
Disadvantage: requires fresh labels as bots evolve; can be evaded
              if attacker knows the feature set.
"""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from data.dataset import BotDetectionDataset


class IsolationForestDetector:
    """
    Unsupervised anomaly detector using Isolation Forest.
    Ideal for catching novel bot patterns without labelled data.
    """

    def __init__(self, cfg):
        self.cfg   = cfg.ml
        self.model = IsolationForest(
            n_estimators=cfg.ml.if_n_estimators,
            contamination=cfg.ml.if_contamination,
            max_samples=cfg.ml.if_max_samples,
            random_state=cfg.ml.if_random_state,
            n_jobs=-1,
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.feature_names: List[str] = []

    def fit(self, dataset: BotDetectionDataset) -> "IsolationForestDetector":
        # Train only on human sessions (normal behaviour)
        human_mask = dataset.y_binary == 0
        X_human    = dataset.X[human_mask]
        self.feature_names = dataset.feature_names

        X_scaled = self.scaler.fit_transform(X_human)
        # Fit on human data only (contamination ~ 0)
        self.model.contamination = 0.01
        self.model.fit(X_scaled)
        self.trained = True
        print(f"  IsolationForest trained on {X_human.shape[0]} human sessions")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores ∈ [0, 1] (1 = most anomalous).
        """
        if not self.trained:
            raise RuntimeError("Call fit() before score()")
        X_scaled = self.scaler.transform(X)
        # sklearn returns decision_function: negative = anomalous
        raw = self.model.decision_function(X_scaled)   # shape (N,)
        # Normalise to [0, 1] — lower decision function → higher anomaly risk
        score = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return score.clip(0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions: 0 = normal, 1 = anomalous."""
        scores = self.score(X)
        return (scores > 0.5).astype(int)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str, cfg) -> "IsolationForestDetector":
        det = cls(cfg)
        with open(path, "rb") as f:
            data = pickle.load(f)
        det.model = data["model"]
        det.scaler = data["scaler"]
        det.feature_names = data["feature_names"]
        det.trained = True
        return det


class GradientBoostingDetector:
    """
    Supervised bot classifier using Gradient Boosting.

    Provides:
      - Binary bot/human classification
      - Calibrated probability scores
      - Feature importance for explainability
    """

    def __init__(self, cfg):
        self.cfg   = cfg.ml
        self.model = GradientBoostingClassifier(
            n_estimators=cfg.ml.gb_n_estimators,
            max_depth=cfg.ml.gb_max_depth,
            learning_rate=cfg.ml.gb_learning_rate,
            subsample=cfg.ml.gb_subsample,
            random_state=cfg.ml.gb_random_state,
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.feature_names: List[str] = []
        self._feature_importances: Optional[np.ndarray] = None

    def fit(self, dataset: BotDetectionDataset) -> "GradientBoostingDetector":
        self.feature_names = dataset.feature_names
        X_scaled = self.scaler.fit_transform(dataset.X)
        print(f"  GradientBoosting training on {dataset.X.shape[0]} sessions "
              f"({dataset.y_binary.sum()} bots)...")
        self.model.fit(X_scaled, dataset.y_binary)
        self._feature_importances = self.model.feature_importances_
        self.trained = True
        train_acc = (self.model.predict(X_scaled) == dataset.y_binary).mean()
        print(f"  Train accuracy: {train_acc:.4f}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return bot probability ∈ [0, 1]."""
        if not self.trained:
            raise RuntimeError("Call fit() before score()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def top_features(self, n: int = 15) -> List[Tuple[str, float]]:
        """Return top-n most important features."""
        if self._feature_importances is None:
            return []
        idx  = np.argsort(self._feature_importances)[::-1][:n]
        return [(self.feature_names[i], float(self._feature_importances[i])) for i in idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names,
                         "importances": self._feature_importances}, f)

    @classmethod
    def load(cls, path: str, cfg) -> "GradientBoostingDetector":
        det = cls(cfg)
        with open(path, "rb") as f:
            data = pickle.load(f)
        det.model = data["model"]
        det.scaler = data["scaler"]
        det.feature_names = data["feature_names"]
        det._feature_importances = data.get("importances")
        det.trained = True
        return det
