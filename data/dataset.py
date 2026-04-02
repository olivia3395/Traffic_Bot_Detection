"""
data/dataset.py — Dataset management: feature matrix construction and splitting.
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional

import numpy as np


class BotDetectionDataset:
    """
    Holds the feature matrix X and labels y for training/evaluation.

    Labels:
        0 = legitimate human traffic
        1 = bot / automated traffic

    Multi-class labels (for detailed analysis):
        0 = human
        1 = simple_bot
        2 = scraper
        3 = cred_stuffer
        4 = llm_agent
    """

    CLASS_MAP = {
        "human": 0, "simple_bot": 1,
        "scraper": 2, "cred_stuffer": 3, "llm_agent": 4,
    }
    BINARY_MAP = {"human": 0}   # everything else → 1

    def __init__(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        y_multi: np.ndarray,
        feature_names: List[str],
        session_ids: List[str],
        traffic_classes: List[str],
    ):
        self.X               = X
        self.y_binary        = y_binary          # 0 / 1
        self.y_multi         = y_multi           # 0-4
        self.feature_names   = feature_names
        self.session_ids     = session_ids
        self.traffic_classes = traffic_classes

    def train_test_split(
        self, test_fraction: float = 0.2, seed: int = 42
    ) -> Tuple["BotDetectionDataset", "BotDetectionDataset"]:
        rng = np.random.RandomState(seed)
        n   = len(self.y_binary)
        idx = rng.permutation(n)
        n_test = int(n * test_fraction)
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]
        return self._subset(train_idx), self._subset(test_idx)

    def _subset(self, idx: np.ndarray) -> "BotDetectionDataset":
        return BotDetectionDataset(
            X=self.X[idx],
            y_binary=self.y_binary[idx],
            y_multi=self.y_multi[idx],
            feature_names=self.feature_names,
            session_ids=[self.session_ids[i] for i in idx],
            traffic_classes=[self.traffic_classes[i] for i in idx],
        )

    def class_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self.traffic_classes))

    def __len__(self) -> int:
        return len(self.y_binary)

    def __repr__(self) -> str:
        dist = self.class_distribution()
        return (
            f"BotDetectionDataset(n={len(self)}, "
            f"features={len(self.feature_names)}, "
            f"classes={dist})"
        )
