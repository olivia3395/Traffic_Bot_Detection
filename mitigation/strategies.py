"""
mitigation/strategies.py — Mitigation strategies based on risk score.

Action hierarchy
────────────────
  ALLOW     [0.00–0.30] : Serve normally. Log for monitoring.
  MONITOR   [0.30–0.55] : Serve but flag for analyst review.
  THROTTLE  [0.55–0.70] : Introduce delay (2s). Rate-limit to 10 rpm.
  CHALLENGE [0.70–0.85] : Serve a JS proof-of-work or CAPTCHA.
  BLOCK     [0.85–1.00] : Return 403. Log IP + fingerprint. Alert.

Adaptive mitigation
────────────────────
The system adapts based on:
  - Business context (checkout page → lower threshold)
  - Time of day (off-hours → higher scrutiny)
  - IP reputation (known proxy ASN → lower threshold)
  - Historical session behaviour (repeat offender → escalate)

False-positive management
──────────────────────────
To protect legitimate users:
  - CHALLENGE before BLOCK (humans pass, most bots fail)
  - Allowlist known good crawlers (Googlebot, Bingbot)
  - Appeal mechanism for blocked users
  - Decay: block expires after N hours
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class Action(str, Enum):
    ALLOW     = "allow"
    MONITOR   = "monitor"
    THROTTLE  = "throttle"
    CHALLENGE = "challenge"
    BLOCK     = "block"


@dataclass
class MitigationDecision:
    action:       Action
    risk_score:   float
    reason:       str
    session_id:   str
    delay_ms:     int = 0
    block_ttl_s:  int = 0
    challenge_type: Optional[str] = None

    def __str__(self) -> str:
        extra = ""
        if self.delay_ms:
            extra += f"  delay={self.delay_ms}ms"
        if self.block_ttl_s:
            extra += f"  block_ttl={self.block_ttl_s}s"
        return (
            f"[{self.action.upper():<9}] "
            f"score={self.risk_score:.3f}  "
            f"session={self.session_id}  "
            f"reason={self.reason[:60]}{extra}"
        )


class MitigationEngine:
    """
    Converts an ensemble risk score into a concrete mitigation action.

    Usage:
        engine = MitigationEngine(cfg)
        decision = engine.decide(score_result, session)
    """

    def __init__(self, cfg):
        self.cfg = cfg.mitigation
        self._block_registry: Dict[str, float] = {}   # ip → block_expiry

    def decide(
        self,
        score_result: Dict,
        session=None,
        page_context: str = "general",
    ) -> MitigationDecision:
        """
        Make a mitigation decision.

        Args:
            score_result  : output from EnsembleDetector.score_session()
            session       : Session object (optional, for IP allowlist check)
            page_context  : "general" | "checkout" | "login" | "search"
                            Sensitive pages have lower block thresholds.

        Returns:
            MitigationDecision
        """
        risk   = score_result["risk_score"]
        sid    = score_result.get("session_id", "unknown")
        reason = score_result.get("explanation", "")

        # ── Allowlist check ────────────────────────────────────────────────
        if session and hasattr(session, "ip"):
            if self._is_allowlisted(session):
                return MitigationDecision(
                    Action.ALLOW, risk, "allowlisted IP/ASN", sid
                )

        # ── Context-sensitive threshold adjustment ─────────────────────────
        # Checkout and login are higher value targets — lower the block threshold
        threshold_delta = {
            "checkout": -0.10,
            "login":    -0.10,
            "search":   +0.05,
            "general":   0.00,
        }.get(page_context, 0.0)
        adj_risk = risk + threshold_delta   # adjusted risk

        # ── Determine action ───────────────────────────────────────────────
        thresholds = self.cfg.thresholds
        if adj_risk >= thresholds["block"][0]:
            return MitigationDecision(
                Action.BLOCK, risk, reason, sid,
                block_ttl_s=self.cfg.block_duration_seconds,
            )
        elif adj_risk >= thresholds["challenge"][0]:
            return MitigationDecision(
                Action.CHALLENGE, risk, reason, sid,
                challenge_type=self.cfg.challenge_type,
            )
        elif adj_risk >= thresholds["throttle"][0]:
            return MitigationDecision(
                Action.THROTTLE, risk, reason, sid,
                delay_ms=int(self.cfg.throttle_delay_seconds * 1000),
            )
        elif adj_risk >= thresholds["monitor"][0]:
            return MitigationDecision(Action.MONITOR, risk, reason, sid)
        else:
            return MitigationDecision(Action.ALLOW, risk, "clean traffic", sid)

    def _is_allowlisted(self, session) -> bool:
        """Check if session IP is on the allowlist."""
        # Placeholder: in production, check against ASN database
        return False

    def action_summary(self, decisions: list) -> Dict[str, int]:
        """Count decisions by action type."""
        from collections import Counter
        return dict(Counter(d.action.value for d in decisions))
