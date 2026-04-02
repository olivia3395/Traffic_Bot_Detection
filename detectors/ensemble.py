"""
detectors/ensemble.py — Weighted ensemble combining all three detector layers.

The ensemble produces a single calibrated risk score [0, 1]
by combining outputs from:
  Layer 1: StatisticalDetector   (fast, rule-based)
  Layer 2a: IsolationForest      (unsupervised ML)
  Layer 2b: GradientBoosting     (supervised ML)
  Layer 3: LLMAgentDetector      (LLM-specific fingerprinting)

Combination strategy
─────────────────────
  1. Weighted average of all four scores (configurable weights)
  2. Hard override: if any single detector is >95% confident → use that score
  3. Agreement amplification: if 3+ detectors agree (all >0.5) → boost score

This multi-layer design gives:
  - Low false-positive rate (consensus required for high-confidence blocks)
  - High recall (any single strong signal triggers investigation)
  - Robustness (no single detector is a bottleneck)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.simulator import Session
from detectors.statistical import StatisticalDetector
from detectors.ml_detector import IsolationForestDetector, GradientBoostingDetector
from detectors.llm_detector import LLMAgentDetector


class EnsembleDetector:
    """
    Multi-layer ensemble bot detector.

    Usage:
        ensemble = EnsembleDetector(cfg, if_det, gb_det, llm_det, stat_det)
        result = ensemble.score_session(session)
    """

    def __init__(
        self,
        cfg,
        if_detector: IsolationForestDetector,
        gb_detector: GradientBoostingDetector,
        llm_detector: LLMAgentDetector,
        stat_detector: StatisticalDetector,
    ):
        self.cfg          = cfg
        self.if_det       = if_detector
        self.gb_det       = gb_detector
        self.llm_det      = llm_detector
        self.stat_det     = stat_detector
        self.ens_cfg      = cfg.ensemble

    def score_session(self, session: Session) -> Dict:
        """
        Score a single session with the full ensemble.

        Returns dict with:
            risk_score        : float [0,1] — final risk score
            statistical_score : float
            isolation_score   : float
            gb_score          : float
            llm_score         : float
            explanation       : str
            n_detectors_firing: int (>0.5 threshold)
        """
        from features.http_features import extract_http_features
        from features.behavioral_features import extract_behavioral_features

        # ── Layer 1: Statistical ─────────────────────────────────────────
        stat_score, stat_reason = self.stat_det.score(session)

        # ── Build feature vector for ML detectors ────────────────────────
        from features.feature_pipeline import extract_all_features
        feats = extract_all_features(session, self.cfg)
        feat_names = list(feats.keys())
        X = np.array([[feats[k] for k in feat_names]], dtype=np.float32)

        # ── Layer 2a: Isolation Forest ────────────────────────────────────
        if self.if_det.trained:
            if_score = float(self.if_det.score(X)[0])
        else:
            if_score = 0.0

        # ── Layer 2b: Gradient Boosting ───────────────────────────────────
        if self.gb_det.trained:
            gb_score = float(self.gb_det.score(X)[0])
        else:
            gb_score = 0.0

        # ── Layer 3: LLM fingerprint ──────────────────────────────────────
        llm_score, llm_signals, llm_conf = self.llm_det.score(session)

        # ── Ensemble combination ──────────────────────────────────────────
        weights = self.ens_cfg.weights
        scores  = {
            "statistical":       stat_score,
            "isolation_forest":  if_score,
            "gradient_boosting": gb_score,
            "llm_detector":      llm_score,
        }

        # Weighted average
        weighted_sum   = sum(weights[k] * v for k, v in scores.items())
        total_weight   = sum(weights.values())
        ensemble_score = weighted_sum / max(total_weight, 1e-9)

        # Hard override: single very high-confidence signal
        max_score = max(scores.values())
        if max_score >= self.ens_cfg.hard_override_threshold:
            ensemble_score = max(ensemble_score, max_score * 0.95)

        # Agreement amplification
        n_firing = sum(1 for v in scores.values() if v > 0.5)
        if n_firing >= self.ens_cfg.min_agreement:
            ensemble_score = min(1.0, ensemble_score * 1.1)

        # Cap to [0, 1]
        ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))

        # ── Build explanation ─────────────────────────────────────────────
        explanation = (
            f"Statistical={stat_score:.3f}[{stat_reason[:40]}] | "
            f"IF={if_score:.3f} | GB={gb_score:.3f} | "
            f"LLM={llm_score:.3f}[{llm_conf}] | "
            f"Ensemble={ensemble_score:.3f} | Firing={n_firing}/4"
        )

        return {
            "risk_score":          ensemble_score,
            "statistical_score":   stat_score,
            "isolation_score":     if_score,
            "gb_score":            gb_score,
            "llm_score":           llm_score,
            "llm_confidence":      llm_conf,
            "llm_signals":         llm_signals,
            "n_detectors_firing":  n_firing,
            "explanation":         explanation,
            "session_id":          session.session_id,
        }

    def score_batch(self, sessions: List[Session]) -> List[Dict]:
        """Score a list of sessions."""
        return [self.score_session(s) for s in sessions]
