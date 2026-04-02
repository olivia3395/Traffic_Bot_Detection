"""
data/csic_loader.py — CSIC 2010 HTTP Dataset loader.

Dataset format
──────────────
CSIC 2010 contains raw HTTP/1.1 requests, one per block separated by
blank lines. Each block looks like:

    GET /tienda1/publico/pagar.jsp?query=alienorde HTTP/1.1
    User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) ...
    Pragma: no-cache
    Cache-Control: no-cache, no-store
    Accept: text/xml,application/xml,application/xhtml+xml,...
    Accept-Encoding: gzip, deflate
    Accept-Charset: ISO-8859-1,utf-8;q=0.7,*;q=0.7
    Host: 127.0.0.1:8080
    Connection: keep-alive
    Content-Length: 0

    POST /tienda1/publico/anadir.jsp HTTP/1.1
    User-Agent: Mozilla/5.0 ...
    Content-Type: application/x-www-form-urlencoded
    Content-Length: 116

    id=3&nombre=Vino+Blanco&precio=11.75&cantidad=...

Labels
──────
  normalTrafficTraining.txt → label = 0 (human / legitimate)
  normalTrafficTest.txt     → label = 0
  anomalousTrafficTest.txt  → label = 1 (bot / attacker)

Session reconstruction
──────────────────────
CSIC 2010 does not have explicit session IDs. We reconstruct sessions
by grouping consecutive requests into pseudo-sessions of fixed size N
(default N=20). This is a standard technique for request-level datasets
without session boundaries.

Mapping to your existing pipeline
──────────────────────────────────
The loader converts each parsed block into a Request object and groups
them into Session objects — exactly the format the existing feature
extraction pipeline expects. No downstream code needs to change.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

from data.simulator import Request, Session, TrafficClass


# ---------------------------------------------------------------------------
# Raw HTTP request parser
# ---------------------------------------------------------------------------

def _parse_http_block(block: str, base_time_ms: float = 1_700_000_000_000.0) -> Optional[Dict]:
    """
    Parse a single raw HTTP request block into a structured dict.

    Args:
        block        : one HTTP request as a raw string
        base_time_ms : simulated start timestamp in milliseconds

    Returns:
        dict with keys: method, url, version, headers, body, user_agent
        or None if the block is malformed / empty.
    """
    lines = block.strip().splitlines()
    if not lines:
        return None

    # ── Request line ──────────────────────────────────────────────────────
    request_line = lines[0].strip()
    parts = request_line.split(" ", 2)
    if len(parts) < 2:
        return None

    method  = parts[0].upper()
    url     = parts[1]
    version = parts[2] if len(parts) > 2 else "HTTP/1.1"

    if method not in ("GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"):
        return None

    # ── Headers ───────────────────────────────────────────────────────────
    headers: Dict[str, str] = {}
    body_lines: List[str]   = []
    in_body = False

    for line in lines[1:]:
        if not in_body and line.strip() == "":
            in_body = True
            continue
        if in_body:
            body_lines.append(line)
        else:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

    body = "\n".join(body_lines).strip()

    return {
        "method":     method,
        "url":        url,
        "version":    version,
        "headers":    headers,
        "body":       body,
        "user_agent": headers.get("user-agent", ""),
    }


# ---------------------------------------------------------------------------
# Request → simulator.Request conversion
# ---------------------------------------------------------------------------

def _to_request(
    parsed: Dict,
    timestamp_ms: float,
    session_id: str,
    ip: str,
    label: int,
) -> Request:
    """Convert a parsed HTTP dict to a simulator.Request object."""
    url = parsed["url"]
    is_api = any(seg in url for seg in ("/api/", "/rest/", "/json/", "/xml/"))

    # Infer status code heuristically from URL patterns
    # (CSIC 2010 doesn't record server responses)
    status = 200
    if label == 1:
        # Attack requests — many would get 400/403/500 in reality
        import random
        status = random.choice([200, 400, 403, 500, 200, 200])

    return Request(
        timestamp_ms=timestamp_ms,
        method=parsed["method"],
        url=url,
        status_code=status,
        response_time_ms=80.0,
        headers=parsed["headers"],
        user_agent=parsed["user_agent"],
        ip=ip,
        session_id=session_id,
        request_size_bytes=len(parsed["body"]) + 200,
        response_size_bytes=5000 + abs(hash(url)) % 50000,
        is_api=is_api,
        referrer=None,
    )


# ---------------------------------------------------------------------------
# File parser → list of raw requests
# ---------------------------------------------------------------------------

def _parse_file(path: str, label: int) -> List[Tuple[Dict, int]]:
    """
    Parse a CSIC 2010 text file into a list of (parsed_request, label).
    """
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        content = f.read()

    # Split on double newlines (blank lines between requests)
    blocks = re.split(r"\n\s*\n", content)
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parsed = _parse_http_block(block)
        if parsed:
            results.append((parsed, label))

    return results


# ---------------------------------------------------------------------------
# Session reconstruction from flat request list
# ---------------------------------------------------------------------------

def _build_sessions(
    requests: List[Tuple[Dict, int]],
    session_size: int = 20,
    iat_ms: float = 2000.0,
    iat_std_ms: float = 800.0,
    ip_prefix: str = "10.0",
    seed: int = 42,
) -> List[Session]:
    """
    Group a flat list of (parsed_request, label) into Session objects.

    Since CSIC 2010 has no session boundaries, we group consecutive
    requests into pseudo-sessions of `session_size` requests each.

    Args:
        requests    : list of (parsed_dict, label) tuples
        session_size: number of requests per pseudo-session
        iat_ms      : mean inter-arrival time within a session (ms)
        iat_std_ms  : std of IAT (ms)
        ip_prefix   : IP prefix for generated IPs

    Returns:
        List of Session objects
    """
    import random
    rng = random.Random(seed)

    sessions: List[Session] = []
    ip_counter = 0

    for i in range(0, len(requests), session_size):
        chunk = requests[i : i + session_size]
        if not chunk:
            break

        # All requests in a chunk share the same label
        # (majority label if mixed — shouldn't happen with proper file splitting)
        labels  = [lbl for _, lbl in chunk]
        label   = max(set(labels), key=labels.count)
        tc      = TrafficClass.HUMAN if label == 0 else TrafficClass.SCRAPER

        # Assign a consistent IP to each pseudo-session
        ip_counter += 1
        ip = f"{ip_prefix}.{ip_counter // 256 % 256}.{ip_counter % 256}"
        sid = hashlib.md5(f"{ip}{i}".encode()).hexdigest()[:12]

        # Reconstruct user agent from first request that has one
        ua = next(
            (p["user_agent"] for p, _ in chunk if p["user_agent"]),
            "Mozilla/5.0 (CSIC-2010)"
        )

        session = Session(
            session_id=sid,
            ip=ip,
            user_agent=ua,
            traffic_class=tc,
            start_time_ms=1_700_000_000_000.0 + i * 5000.0,
            label=label,
        )

        # Assign timestamps
        t = session.start_time_ms
        for j, (parsed, _) in enumerate(chunk):
            if j > 0:
                iat = max(100.0, rng.gauss(iat_ms, iat_std_ms))
                t  += iat
            req = _to_request(parsed, t, sid, ip, label)
            session.requests.append(req)

        sessions.append(session)

    return sessions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csic_dataset(
    data_dir: str,
    normal_train_file: str  = "normalTrafficTraining.txt",
    normal_test_file: str   = "normalTrafficTest.txt",
    anomalous_file: str     = "anomalousTrafficTest.txt",
    session_size: int       = 20,
    max_normal: Optional[int] = None,
    max_anomalous: Optional[int] = None,
    seed: int               = 42,
) -> List[Session]:
    """
    Load and parse the CSIC 2010 dataset into Session objects.

    Args:
        data_dir          : directory containing the three .txt files
        normal_train_file : filename for normal training traffic
        normal_test_file  : filename for normal test traffic
        anomalous_file    : filename for attack traffic
        session_size      : requests per pseudo-session
        max_normal        : cap on normal sessions (None = all)
        max_anomalous     : cap on anomalous sessions (None = all)
        seed              : random seed for reproducibility

    Returns:
        List of Session objects, ready for feature extraction.

    Raises:
        FileNotFoundError if any of the expected files are missing.

    Usage:
        sessions = load_csic_dataset("data/csic2010/")
        dataset  = build_dataset(sessions, cfg)
    """
    print("\n[CSIC 2010 Dataset Loader]")

    # ── Check files ──────────────────────────────────────────────────────
    files = {
        "normal_train": os.path.join(data_dir, normal_train_file),
        "normal_test":  os.path.join(data_dir, normal_test_file),
        "anomalous":    os.path.join(data_dir, anomalous_file),
    }
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CSIC 2010 file not found: {path}\n"
                f"Download from:\n"
                f"  https://www.kaggle.com/datasets/victorsolano/http-dataset-csic-2010\n"
                f"  or: http://www.isi.csic.es/dataset/\n"
                f"Then place the .txt files in: {data_dir}"
            )

    # ── Parse normal traffic ──────────────────────────────────────────────
    normal_reqs = []
    for key in ("normal_train", "normal_test"):
        path   = files[key]
        parsed = _parse_file(path, label=0)
        normal_reqs.extend(parsed)
        print(f"  {os.path.basename(path):<40} {len(parsed):>7,} requests  (label=0, normal)")

    # ── Parse anomalous traffic ───────────────────────────────────────────
    anomalous_reqs = _parse_file(files["anomalous"], label=1)
    print(f"  {anomalous_file:<40} {len(anomalous_reqs):>7,} requests  (label=1, attack)")

    # ── Apply caps ────────────────────────────────────────────────────────
    import random
    rng = random.Random(seed)
    rng.shuffle(normal_reqs)
    rng.shuffle(anomalous_reqs)

    if max_normal:
        normal_reqs = normal_reqs[:max_normal * session_size]
    if max_anomalous:
        anomalous_reqs = anomalous_reqs[:max_anomalous * session_size]

    # ── Build sessions ───────────────────────────────────────────────────
    print(f"\n  Building pseudo-sessions (session_size={session_size})...")
    normal_sessions    = _build_sessions(normal_reqs,    session_size, ip_prefix="10.1", seed=seed)
    anomalous_sessions = _build_sessions(anomalous_reqs, session_size,
                                          iat_ms=500.0, iat_std_ms=100.0,
                                          ip_prefix="10.2", seed=seed+1)

    print(f"  Normal sessions    : {len(normal_sessions):,}")
    print(f"  Anomalous sessions : {len(anomalous_sessions):,}")

    # ── Combine and shuffle ───────────────────────────────────────────────
    all_sessions = normal_sessions + anomalous_sessions
    rng.shuffle(all_sessions)

    n_bot = sum(1 for s in all_sessions if s.label == 1)
    n_hum = sum(1 for s in all_sessions if s.label == 0)
    print(f"  Total sessions     : {len(all_sessions):,}  "
          f"(normal={n_hum:,}, attack={n_bot:,}, "
          f"ratio={n_bot/max(len(all_sessions),1)*100:.1f}% attack)")

    return all_sessions


def attack_type_breakdown(data_dir: str, anomalous_file: str = "anomalousTrafficTest.txt") -> Dict[str, int]:
    """
    Analyse the types of attacks in the anomalous traffic file.

    CSIC 2010 attack patterns detected by URL signature:
      SQL injection      : UNION, SELECT, DROP, INSERT, --, /*
      XSS                : <script>, javascript:, onload=
      Path traversal     : ../../, /etc/passwd
      CSRF               : cross-site form submissions
      Command injection  : ;ls, |cat, &&
      Buffer overflow    : very long parameter values (>500 chars)

    Returns dict of attack_type → count.
    """
    from collections import Counter
    path    = os.path.join(data_dir, anomalous_file)
    parsed  = _parse_file(path, label=1)
    counter: Counter = Counter()

    sql_re   = re.compile(r"(union|select|drop|insert|delete|--|/\*|'\s*or)", re.I)
    xss_re   = re.compile(r"(<script|javascript:|onerror=|onload=|alert\()", re.I)
    trav_re  = re.compile(r"(\.\./|/etc/|/proc/|/windows/)", re.I)
    cmd_re   = re.compile(r"(;ls|;cat|&&|`|%0a|%0d)", re.I)
    buf_re   = re.compile(r"[^&=]{500,}")

    for p, _ in parsed:
        url_and_body = p["url"] + " " + p.get("body", "")
        if sql_re.search(url_and_body):
            counter["sql_injection"] += 1
        elif xss_re.search(url_and_body):
            counter["xss"] += 1
        elif trav_re.search(url_and_body):
            counter["path_traversal"] += 1
        elif cmd_re.search(url_and_body):
            counter["command_injection"] += 1
        elif buf_re.search(url_and_body):
            counter["buffer_overflow"] += 1
        else:
            counter["other_anomaly"] += 1

    return dict(counter)
