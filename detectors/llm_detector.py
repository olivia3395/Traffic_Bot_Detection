"""
detectors/llm_detector.py — LLM-agent specific detector (Layer 3).

This detector is tuned specifically for LLM-orchestrated browsing agents.
It combines the fingerprint signals from features/llm_fingerprints.py
into a detection decision with calibrated confidence.

Why a dedicated LLM-agent detector?
──────────────────────────────────────
Standard bot detectors are tuned for:
  - High request rates (simple bots)
  - Missing headers (naive scrapers)
  - Credential stuffing patterns

LLM agents are different:
  - Request rates are MODERATE (~0.5–2 req/s) — not obviously high
  - Headers are COMPLETE (using Playwright/headless Chrome)
  - Sessions are DEEP but purposeful

They slip through standard detectors because individually, each signal
is within "normal" range. Only the combination of:
  (1) moderate rate + (2) extreme regularity + (3) systematic coverage
  + (4) API probing + (5) very deep session
... creates the distinctive LLM fingerprint.
"""

from __future__ import annotations
from typing import Dict, Tuple

from data.simulator import Session
from features.llm_fingerprints import compute_llm_fingerprint


class LLMAgentDetector:
    """
    Dedicated detector for LLM-powered browsing agents.

    Produces:
      - llm_risk_score ∈ [0, 1]
      - Per-signal breakdown for explainability
      - Confidence level (LOW / MEDIUM / HIGH)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, session: Session) -> Tuple[float, Dict[str, float], str]:
        """
        Compute LLM-agent risk score.

        Returns:
            (risk_score, signal_breakdown, confidence_label)
        """
        if len(session.requests) < self.cfg.features.min_requests_to_score:
            return 0.0, {}, "INSUFFICIENT_DATA"

        fingerprint = compute_llm_fingerprint(session, self.cfg)
        llm_score   = fingerprint["llm_score"]

        # Calibrate confidence based on session length
        n = len(session.requests)
        if n < 10:
            confidence = "LOW"
        elif n < 30:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

        return round(llm_score, 4), fingerprint, confidence

    def predict(self, session: Session) -> int:
        """Binary: 0 = not LLM agent, 1 = LLM agent detected."""
        score, _, _ = self.score(session)
        return 1 if score >= 0.50 else 0

    def explain(self, session: Session) -> str:
        """Return human-readable explanation of the LLM detection decision."""
        score, signals, confidence = self.score(session)
        lines = [
            f"LLM Agent Detection — score={score:.3f}  confidence={confidence}",
            f"  Session depth: {len(session.requests)} requests",
        ]
        for signal, value in signals.items():
            if signal == "llm_score":
                continue
            indicator = "⚠" if value > 0.5 else "✓"
            lines.append(f"  {indicator} {signal:<25} {value:.3f}")
        verdict = "🤖 LLM AGENT DETECTED" if score >= 0.5 else "✅ NOT LLM AGENT"
        lines.append(f"\n  {verdict}")
        return "\n".join(lines)
