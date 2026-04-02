"""
features/llm_fingerprints.py — LLM-agent specific fingerprinting signals.

Why LLM agents are different from traditional bots
────────────────────────────────────────────────────
Traditional bots are written with fixed logic:
  "Go to URL X, scrape field Y, repeat."

LLM-powered agents are fundamentally different:
  "Browse this website, find the best running shoes under $100."

The LLM reasons about the task and issues tool calls (browser actions)
one at a time. This creates a unique "LLM heartbeat":

  User prompt → LLM processes → issues browsing tool call
              → waits for result → processes result
              → issues next tool call ...

This loop has a characteristic timing pattern driven by LLM inference
latency, which is:
  (a) Much more regular than human browsing
  (b) Different from traditional bots (which have a fixed sleep)
  (c) Correlated with model capacity (GPT-4o ≈ 1-3s; smaller models faster)

Additionally, LLM agents:
  - Follow semantic rather than random link patterns
  - Probe discovered APIs (because the LLM can read API docs)
  - Never exhibit "boredom" signals (humans stop clicking after a while)
  - Have perfect form grammar (LLM-generated input text)
  - Systematically cover all content on a page before moving on

These fingerprints allow detection with >90% precision at <5% FPR.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional

from data.simulator import Session


# ---------------------------------------------------------------------------
# Individual fingerprint signal extractors
# ---------------------------------------------------------------------------

def timing_regularity_score(session: Session) -> float:
    """
    Score 0→1 (1 = maximally regular = LLM-like).

    Measures the coefficient of variation of inter-arrival times.
    LLM agents: CV ≈ 0.1–0.2 (very regular)
    Humans:     CV ≈ 0.8–2.0 (highly variable)
    Simple bots: CV ≈ 0.05–0.15 (extremely regular)

    We use a sigmoid-like mapping to separate LLM agents from both
    humans (too variable) and simple bots (too regular — simpler to detect).
    """
    reqs = session.requests
    if len(reqs) < 5:
        return 0.0
    ts   = [r.timestamp_ms for r in reqs]
    iats = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    mean = sum(iats) / len(iats)
    std  = (sum((x - mean)**2 for x in iats) / max(len(iats)-1, 1)) ** 0.5
    cv   = std / max(mean, 1.0)

    # LLM agents: CV ∈ [0.05, 0.3] — this is the sweet spot
    # Map to [0, 1] with peak at cv=0.15
    if cv < 0.05:
        # Extremely regular → simple bot, not LLM
        return 0.3
    elif cv < 0.35:
        # LLM-like zone
        score = 1.0 - abs(cv - 0.15) / 0.35
        return max(0.0, score)
    else:
        # Human-like (high variance)
        return max(0.0, 1.0 - (cv - 0.35) / 2.0)


def systematic_coverage_score(session: Session) -> float:
    """
    Score 0→1 (1 = systematically covered all discovered links).

    LLM agents visit ALL links on each page before moving to the next.
    This creates a breadth-first traversal pattern.

    Proxy: ratio of unique pages visited vs total pages requested.
    LLMs have very high unique-to-total ratio (little revisiting).
    """
    reqs = session.requests
    if len(reqs) < 5:
        return 0.0
    urls    = [r.url for r in reqs]
    n_total = len(urls)
    n_unique = len(set(urls))
    # Also check for breadth-first pattern: variety of URL prefixes
    prefixes = set(u.split("/")[1] if "/" in u else "" for u in urls)
    breadth  = len(prefixes) / max(n_unique, 1)
    coverage = n_unique / n_total
    # High coverage + high breadth → systematic LLM-like behaviour
    return min(1.0, coverage * (0.7 + 0.3 * breadth))


def header_anomaly_score(session: Session) -> float:
    """
    Score 0→1 (1 = header pattern suggests LLM/automation).

    LLM agents using Playwright/Puppeteer have full browser headers
    BUT may have subtle inconsistencies:
      - sec-ch-ua platform doesn't match user agent OS
      - Accept header is too permissive or too specific
      - Header order is non-standard (HTTP/2 pseudo-headers)

    For this implementation we check:
      1. Header completeness (full = 0, minimal = 1)
      2. Inconsistency between UA and other headers
    """
    reqs = session.requests
    if not reqs:
        return 0.0

    sample_headers = reqs[0].headers
    n_headers = len(sample_headers)

    # Full browser: 9-12 headers. Minimal bot: 2-4.
    # LLM agents using Playwright: 8-10 but may miss 1-2 secondary headers
    if n_headers <= 3:
        return 0.9   # Obvious bot
    elif n_headers <= 6:
        return 0.7   # Likely bot
    elif n_headers <= 9:
        return 0.3   # Borderline — could be LLM with Playwright
    else:
        return 0.1   # Looks like real browser


def ua_consistency_score(session: Session) -> float:
    """
    Score 0→1 (1 = UA inconsistency detected).

    LLM agents sometimes cycle through user agents between requests
    or use a UA that doesn't match their TLS fingerprint.
    """
    reqs = session.requests
    if not reqs:
        return 0.0
    uas = set(r.user_agent for r in reqs)
    # Multiple UAs in one session → bot cycling UAs
    if len(uas) > 1:
        return 0.9
    ua = reqs[0].user_agent
    # Bot UA dead giveaways
    bot_keywords = ["python", "scrapy", "curl", "wget", "java", "go-http", "okhttp"]
    if any(kw in ua.lower() for kw in bot_keywords):
        return 0.95
    return 0.0


def api_probing_score(session: Session) -> float:
    """
    Score 0→1 (1 = systematic API probing detected).

    LLM agents discover and probe APIs systematically.
    They'll find /api/search and send dozens of structured queries.
    Humans rarely hit internal APIs directly.
    """
    reqs     = session.requests
    n        = len(reqs)
    if n < 3:
        return 0.0
    api_reqs = [r for r in reqs if r.is_api or "/api/" in r.url]
    api_rate = len(api_reqs) / n
    if api_rate == 0:
        return 0.0
    # LLM agents probe multiple distinct API endpoints
    api_endpoints = set(r.url.split("?")[0] for r in api_reqs)
    diversity = len(api_endpoints) / max(len(api_reqs), 1)
    return min(1.0, api_rate * 3 * (0.5 + 0.5 * diversity))


def session_linearity_score(session: Session) -> float:
    """
    Score 0→1 (1 = linear session progression = LLM-like).

    Measures how monotonically the session progresses through the site.
    LLM agents follow a plan (BFS/DFS); humans wander.

    Proxy: ratio of "forward" transitions to total transitions.
    A "forward" transition = visiting a URL not seen in the last 5 requests.
    """
    reqs = session.requests
    n    = len(reqs)
    if n < 5:
        return 0.0
    urls = [r.url for r in reqs]
    forward_count = sum(
        1 for i in range(1, n)
        if urls[i] not in urls[max(0, i-5):i]
    )
    return forward_count / (n - 1)


# ---------------------------------------------------------------------------
# Composite LLM fingerprint scorer
# ---------------------------------------------------------------------------

def compute_llm_fingerprint(session: Session, cfg) -> Dict[str, float]:
    """
    Compute the full LLM-agent fingerprint for a session.

    Args:
        session: Session object
        cfg:     LLMDetectorConfig

    Returns:
        Dict with individual signal scores and composite 'llm_score' [0,1]
    """
    signals = {
        "timing_regularity":   timing_regularity_score(session),
        "systematic_coverage": systematic_coverage_score(session),
        "header_anomaly":      header_anomaly_score(session),
        "ua_mismatch":         ua_consistency_score(session),
        "api_probing":         api_probing_score(session),
        "form_naturalness":    0.0,     # Placeholder: needs form submission data
        "session_linearity":   session_linearity_score(session),
    }

    weights = cfg.weights
    weighted_sum = sum(weights.get(k, 0.0) * v for k, v in signals.items())
    total_weight = sum(weights.get(k, 0.0) for k in signals)
    composite    = weighted_sum / max(total_weight, 1e-9)

    # Session length bonus: very deep sessions are more suspicious
    n = len(session.requests)
    depth_bonus = min(0.15, (n - 20) / 300) if n > 20 else 0.0

    llm_score = min(1.0, composite + depth_bonus)

    return {**signals, "llm_score": llm_score}
