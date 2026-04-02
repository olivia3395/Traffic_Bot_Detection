"""
features/behavioral_features.py — Session-level behavioural feature extraction.

These features capture *how* a session navigates, not just *what* it requests.
Behavioural patterns are much harder for sophisticated bots to fake because
they require understanding human browsing psychology.

Key signals
───────────
  Navigation graph  : are pages visited in a coherent order?
  Exploration ratio : does the user explore or drill straight down?
  Backtrack rate    : humans frequently go back; bots rarely do
  Dwell time proxy  : estimated time spent on each page
  Content diversity : variety of content types accessed
  Momentum          : does the session accelerate (bot) or decelerate (human tired)?
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List

from data.simulator import Session


def extract_behavioral_features(session: Session) -> Dict[str, float]:
    """
    Extract behavioural features from a session's request sequence.

    Args:
        session: Session with ordered Request objects

    Returns:
        Dict of feature_name → float
    """
    reqs = session.requests
    n    = len(reqs)
    if n < 2:
        return _zero_behavioral_features()

    urls       = [r.url for r in reqs]
    timestamps = [r.timestamp_ms for r in reqs]

    # ── Navigation linearity ──────────────────────────────────────────────
    # Bots (especially LLM agents) follow a linear path through the site.
    # Humans wander. We measure how "tree-like" the navigation graph is.
    url_sequence   = urls
    unique_urls    = list(dict.fromkeys(url_sequence))   # preserve order, dedupe
    n_unique       = len(unique_urls)
    revisit_ratio  = 1.0 - (n_unique / n)               # humans revisit pages

    # ── Backtracking ──────────────────────────────────────────────────────
    # A backtrack = current URL was visited in the last 3 pages
    backtrack_count = sum(
        1 for i in range(3, n)
        if urls[i] in urls[max(0, i-3):i-1]
    )
    backtrack_rate = backtrack_count / max(n - 3, 1)

    # ── Dwell time (proxy) ────────────────────────────────────────────────
    # Estimated time spent on each page = next_timestamp - current_timestamp
    # Humans spend more time on content pages; bots have uniform dwell times
    page_dwell_times = [timestamps[i+1] - timestamps[i] for i in range(n-1)]
    dwell_mean = sum(page_dwell_times) / len(page_dwell_times)
    dwell_std  = (sum((t - dwell_mean)**2 for t in page_dwell_times) / max(len(page_dwell_times)-1, 1)) ** 0.5
    dwell_cv   = dwell_std / max(dwell_mean, 1.0)    # high CV → human-like

    # Ratio of very short dwell times (< 500ms) — bots load and move on
    short_dwell_ratio = sum(1 for t in page_dwell_times if t < 500) / len(page_dwell_times)

    # ── Path entropy ─────────────────────────────────────────────────────
    # High entropy = exploring many different sections (human-like)
    # Low entropy  = focused on one section (bot-like)
    path_parts = [u.split("/")[1] if "/" in u else u for u in urls]
    section_counts = Counter(path_parts)
    total = sum(section_counts.values())
    path_entropy = -sum(
        (c/total) * math.log2(c/total + 1e-12)
        for c in section_counts.values()
    )

    # ── Session momentum ──────────────────────────────────────────────────
    # Split session into first half and second half
    # Humans slow down as they focus on something interesting
    # Bots maintain constant rate
    half = n // 2
    if half > 1:
        iats = [timestamps[i+1] - timestamps[i] for i in range(n-1)]
        first_half_iat  = sum(iats[:half]) / half
        second_half_iat = sum(iats[half:]) / max(len(iats) - half, 1)
        # Positive → slowing down (human-like), negative → speeding up (bot-like)
        momentum = (second_half_iat - first_half_iat) / max(first_half_iat, 1.0)
    else:
        momentum = 0.0

    # ── Content type diversity ────────────────────────────────────────────
    url_types = {
        "product":  sum(1 for u in urls if "product" in u),
        "category": sum(1 for u in urls if "products" in u or "category" in u),
        "api":      sum(1 for u in urls if "/api/" in u),
        "account":  sum(1 for u in urls if "account" in u or "login" in u),
        "static":   sum(1 for u in urls if "static" in u or ".js" in u or ".css" in u),
        "other":    0,
    }
    url_types["other"] = n - sum(url_types.values())
    content_type_counts = [v for v in url_types.values() if v > 0]
    content_diversity   = len(content_type_counts) / len(url_types)

    # ── Referrer chain coherence ──────────────────────────────────────────
    # LLM agents always set a referrer (they follow links systematically)
    # Humans sometimes type URLs directly, use bookmarks, etc.
    referrers = [r.referrer for r in reqs]
    referrer_chain_rate = sum(1 for ref in referrers if ref) / n

    # ── API probing behaviour ─────────────────────────────────────────────
    api_urls     = [u for u in urls if "/api/" in u]
    api_rate     = len(api_urls) / n
    unique_api   = len(set(api_urls))
    api_variety  = unique_api / max(len(api_urls), 1)

    # ── Error tolerance ───────────────────────────────────────────────────
    # Humans stop after errors; bots continue regardless
    statuses     = [r.status_code for r in reqs]
    error_positions = [i for i, s in enumerate(statuses) if s >= 400]
    # Requests after last error / total requests after first error
    if error_positions:
        last_error_pos = max(error_positions)
        continued_after_error = (n - 1 - last_error_pos) / n
    else:
        continued_after_error = 0.0

    return {
        # Navigation
        "revisit_ratio":             revisit_ratio,
        "backtrack_rate":            backtrack_rate,
        "n_unique_pages":            float(n_unique),
        "path_entropy":              path_entropy,
        # Dwell time
        "dwell_mean_ms":             dwell_mean,
        "dwell_std_ms":              dwell_std,
        "dwell_cv":                  dwell_cv,
        "short_dwell_ratio":         short_dwell_ratio,
        # Session dynamics
        "momentum":                  momentum,
        "content_diversity":         content_diversity,
        # Referrer
        "referrer_chain_rate":       referrer_chain_rate,
        # API behaviour
        "api_rate":                  api_rate,
        "api_variety":               api_variety,
        # Error handling
        "continued_after_error":     continued_after_error,
    }


def _zero_behavioral_features() -> Dict[str, float]:
    keys = [
        "revisit_ratio", "backtrack_rate", "n_unique_pages", "path_entropy",
        "dwell_mean_ms", "dwell_std_ms", "dwell_cv", "short_dwell_ratio",
        "momentum", "content_diversity", "referrer_chain_rate",
        "api_rate", "api_variety", "continued_after_error",
    ]
    return {k: 0.0 for k in keys}
