"""
features/http_features.py — HTTP request-level feature extraction.

Features extracted per session from raw HTTP logs.

Feature groups
──────────────
  Request rate   : rpm, peak_rpm, burst_ratio
  Timing         : iat_mean, iat_std, iat_cv, iat_min, iat_max
  Status codes   : error_rate, rate_404, rate_4xx, rate_5xx
  URLs           : unique_url_ratio, url_entropy, api_ratio, depth_max
  Headers        : header_completeness, has_sec_fetch, has_sec_ch_ua
  Methods        : post_ratio, get_ratio
  Payload sizes  : req_size_mean, resp_size_mean, resp_size_cv
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List

from data.simulator import Session


def _entropy(values: List) -> float:
    """Shannon entropy of a list of values."""
    if not values:
        return 0.0
    counts = Counter(values)
    total  = len(values)
    return -sum(
        (c / total) * math.log2(c / total + 1e-12)
        for c in counts.values()
    )


def extract_http_features(session: Session) -> Dict[str, float]:
    """
    Extract HTTP-level features from a session.

    Args:
        session: Session with a list of Request objects

    Returns:
        Dict of feature_name → float value
    """
    reqs = session.requests
    if not reqs:
        return _zero_http_features()

    n = len(reqs)
    timestamps = [r.timestamp_ms for r in reqs]
    duration_s = max((timestamps[-1] - timestamps[0]) / 1000.0, 0.001)

    # ── Inter-arrival times ───────────────────────────────────────────────
    iats = [timestamps[i+1] - timestamps[i] for i in range(n-1)] if n > 1 else [0.0]
    iat_mean = sum(iats) / len(iats)
    iat_std  = (sum((x - iat_mean)**2 for x in iats) / max(len(iats)-1, 1)) ** 0.5
    iat_cv   = iat_std / max(iat_mean, 1e-6)
    iat_min  = min(iats)
    iat_max  = max(iats)

    # ── Request rates ─────────────────────────────────────────────────────
    rpm_mean = (n / duration_s) * 60.0

    # Burst: fraction of IATs below 100ms (very fast consecutive requests)
    burst_ratio = sum(1 for t in iats if t < 100) / max(len(iats), 1)

    # ── Status codes ──────────────────────────────────────────────────────
    statuses = [r.status_code for r in reqs]
    n_2xx = sum(1 for s in statuses if 200 <= s < 300)
    n_4xx = sum(1 for s in statuses if 400 <= s < 500)
    n_404 = sum(1 for s in statuses if s == 404)
    n_5xx = sum(1 for s in statuses if s >= 500)
    error_rate = (n_4xx + n_5xx) / n
    rate_4xx   = n_4xx / n
    rate_404   = n_404 / n
    rate_5xx   = n_5xx / n

    # ── URL diversity ─────────────────────────────────────────────────────
    urls           = [r.url for r in reqs]
    unique_urls    = set(urls)
    unique_url_ratio = len(unique_urls) / n
    url_entropy    = _entropy(urls)
    api_ratio      = sum(1 for r in reqs if r.is_api) / n
    url_depths     = [len(u.rstrip("/").split("/")) - 1 for u in urls]
    depth_max      = max(url_depths) if url_depths else 0
    depth_mean     = sum(url_depths) / max(len(url_depths), 1)

    # ── HTTP headers ──────────────────────────────────────────────────────
    n_headers_list = [len(r.headers) for r in reqs]
    header_mean    = sum(n_headers_list) / n
    # Full browser sends 9–12 headers; bots typically send 2–4
    header_completeness = min(header_mean / 10.0, 1.0)
    has_sec_fetch = float(any(
        "sec-fetch-dest" in r.headers for r in reqs
    ))
    has_sec_ch_ua = float(any(
        "sec-ch-ua" in r.headers for r in reqs
    ))
    has_accept_lang = float(any(
        "accept-language" in r.headers for r in reqs
    ))

    # ── HTTP methods ──────────────────────────────────────────────────────
    methods    = [r.method for r in reqs]
    post_ratio = methods.count("POST") / n
    get_ratio  = methods.count("GET")  / n

    # ── Payload sizes ─────────────────────────────────────────────────────
    req_sizes  = [r.request_size_bytes  for r in reqs]
    resp_sizes = [r.response_size_bytes for r in reqs]
    req_size_mean  = sum(req_sizes)  / n
    resp_size_mean = sum(resp_sizes) / n
    resp_size_std  = (sum((x - resp_size_mean)**2 for x in resp_sizes) / max(n-1, 1)) ** 0.5
    resp_size_cv   = resp_size_std / max(resp_size_mean, 1.0)

    # ── Referrer chain ────────────────────────────────────────────────────
    referrer_ratio = sum(1 for r in reqs if r.referrer) / n

    # ── Session-level ─────────────────────────────────────────────────────
    session_depth = n
    session_duration_s = duration_s

    return {
        # Timing
        "iat_mean_ms":         iat_mean,
        "iat_std_ms":          iat_std,
        "iat_cv":              iat_cv,
        "iat_min_ms":          iat_min,
        "iat_max_ms":          iat_max,
        # Rate
        "rpm_mean":            rpm_mean,
        "burst_ratio":         burst_ratio,
        # Status codes
        "error_rate":          error_rate,
        "rate_4xx":            rate_4xx,
        "rate_404":            rate_404,
        "rate_5xx":            rate_5xx,
        # URLs
        "unique_url_ratio":    unique_url_ratio,
        "url_entropy":         url_entropy,
        "api_ratio":           api_ratio,
        "depth_max":           float(depth_max),
        "depth_mean":          depth_mean,
        # Headers
        "header_completeness": header_completeness,
        "has_sec_fetch":       has_sec_fetch,
        "has_sec_ch_ua":       has_sec_ch_ua,
        "has_accept_language": has_accept_lang,
        "header_mean_count":   header_mean,
        # Methods
        "post_ratio":          post_ratio,
        "get_ratio":           get_ratio,
        # Payloads
        "req_size_mean":       req_size_mean,
        "resp_size_mean":      resp_size_mean,
        "resp_size_cv":        resp_size_cv,
        # Referrer
        "referrer_ratio":      referrer_ratio,
        # Session
        "session_depth":       float(session_depth),
        "session_duration_s":  session_duration_s,
    }


def _zero_http_features() -> Dict[str, float]:
    return {k: 0.0 for k in extract_http_features.__code__.co_consts if isinstance(k, str)}
