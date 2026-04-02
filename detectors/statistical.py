"""
detectors/statistical.py — Statistical and rule-based anomaly detection.

This is Layer 1: the fastest detection layer.
Runs in microseconds per session. No training required.
Designed to catch obvious bots immediately, freeing ML layers
to focus on sophisticated threats.

Decisions are deterministic and auditable — important for compliance
and for explaining blocks to challenged users.
"""

from __future__ import annotations
from typing import Dict, Tuple

from data.simulator import Session
from features.http_features import extract_http_features


class StatisticalDetector:
    """
    Rule-based detector using statistical thresholds.

    Produces a risk score [0, 1] and a human-readable explanation
    of which rules fired.
    """

    def __init__(self, cfg):
        self.cfg = cfg.statistical

    def score(self, session: Session) -> Tuple[float, str]:
        """
        Score a session using statistical rules.

        Returns:
            (risk_score, explanation)
            risk_score ∈ [0, 1]
        """
        feats   = extract_http_features(session)
        reasons = []
        score   = 0.0

        # ── Rule 1: Request rate ─────────────────────────────────────────
        rpm = feats["rpm_mean"]
        if rpm >= self.cfg.rate_block_rpm:
            score   = max(score, 0.95)
            reasons.append(f"rpm={rpm:.0f} (hard block threshold {self.cfg.rate_block_rpm})")
        elif rpm >= self.cfg.rate_warn_rpm:
            partial = min(0.70, 0.40 + 0.30 * (rpm - self.cfg.rate_warn_rpm) / self.cfg.rate_warn_rpm)
            score   = max(score, partial)
            reasons.append(f"rpm={rpm:.0f} (warn threshold {self.cfg.rate_warn_rpm})")

        # ── Rule 2: IAT regularity ────────────────────────────────────────
        iat_cv = feats["iat_cv"]
        if iat_cv < self.cfg.iat_cv_bot_threshold:
            partial = 0.75 * (1.0 - iat_cv / self.cfg.iat_cv_bot_threshold)
            score   = max(score, partial)
            reasons.append(f"iat_cv={iat_cv:.3f} (too regular, threshold={self.cfg.iat_cv_bot_threshold})")

        # ── Rule 3: 404 error rate ────────────────────────────────────────
        if feats["rate_404"] > 0.15:
            partial = min(0.80, 0.5 + feats["rate_404"])
            score   = max(score, partial)
            reasons.append(f"404_rate={feats['rate_404']:.2f}")

        # ── Rule 4: Missing browser headers ──────────────────────────────
        if feats["has_sec_fetch"] == 0.0 and feats["has_accept_language"] == 0.0:
            score   = max(score, 0.70)
            reasons.append("missing critical browser headers (sec-fetch, accept-language)")

        # ── Rule 5: Bot user agent ────────────────────────────────────────
        ua = session.user_agent.lower()
        bot_keywords = ["python", "scrapy", "curl", "wget", "java", "go-http", "okhttp"]
        if any(kw in ua for kw in bot_keywords):
            score   = max(score, 0.97)
            reasons.append(f"bot user-agent detected: {session.user_agent[:50]}")

        # ── Rule 6: Very deep session ─────────────────────────────────────
        depth = feats["session_depth"]
        if depth > self.cfg.max_human_depth:
            partial = min(0.80, 0.5 + (depth - self.cfg.max_human_depth) / 200.0)
            score   = max(score, partial)
            reasons.append(f"session_depth={depth:.0f} (>{self.cfg.max_human_depth})")

        # ── Rule 7: Burst ratio ────────────────────────────────────────────
        if feats["burst_ratio"] > 0.5:
            score   = max(score, 0.75)
            reasons.append(f"burst_ratio={feats['burst_ratio']:.2f} (>0.5)")

        explanation = "CLEAN" if not reasons else " | ".join(reasons)
        return round(score, 4), explanation

    def predict(self, session: Session) -> int:
        """Return binary prediction: 0 = human, 1 = bot."""
        score, _ = self.score(session)
        return 1 if score >= 0.5 else 0
