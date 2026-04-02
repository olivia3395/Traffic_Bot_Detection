"""
data/simulator.py — Realistic traffic simulator for bot detection research.

Generates synthetic HTTP sessions for five traffic classes:

  1. Human         — organic browsing with natural variance
  2. SimpleBot     — naive scrapers (regular timing, missing headers)
  3. Scraper       — sophisticated crawlers (rotating UAs, moderate timing)
  4. CredStuffer   — credential stuffing (high POST rate, few pages)
  5. LLMAgent      — LLM-orchestrated browsing (the newest threat class)

LLM-Agent Behavioural Model
────────────────────────────
LLM-powered agents (GPT-4o, Claude, Gemini with web browsing) display
unique behavioural signatures that distinguish them from both humans and
traditional bots:

  Timing:     Regular inter-arrival times (LLM "thinking" is consistent)
              IAT coefficient of variation (CV) ≈ 0.1–0.2
              vs humans: CV ≈ 0.8–2.0

  Navigation: Systematic link following (breadth-first or depth-first)
              Visits ALL links on a page before moving on
              No backtracking, no random exploration

  Depth:      Very deep sessions (20–150 pages per session)
              Linear progression through the site structure

  Content:    Reads product descriptions, reviews, FAQs systematically
              Same dwell time regardless of content length

  API:        Discovers and probes internal APIs (autocomplete, search)
              Sends structured, well-formed queries

  Forms:      Inputs are grammatically perfect natural language
              No typos, no corrections, no partial fills
"""

from __future__ import annotations

import random
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class TrafficClass(str, Enum):
    HUMAN          = "human"
    SIMPLE_BOT     = "simple_bot"
    SCRAPER        = "scraper"
    CRED_STUFFER   = "cred_stuffer"
    LLM_AGENT      = "llm_agent"


BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

BOT_UAS = [
    "python-requests/2.31.0",
    "Scrapy/2.11.0 (+https://scrapy.org)",
    "curl/8.4.0",
    "Go-http-client/1.1",
    "Java/11.0.21",
]

STORE_PAGES = [
    "/", "/products", "/products/shoes", "/products/clothing",
    "/products/electronics", "/sale", "/brands", "/search",
    "/product/1001", "/product/1002", "/product/1003",
    "/product/1004", "/product/1005", "/product/1006",
    "/cart", "/checkout", "/account/login", "/account/register",
    "/faq", "/contact", "/about", "/shipping", "/returns",
    "/blog", "/blog/style-guide", "/blog/size-guide",
    "/api/search?q=shoes", "/api/autocomplete?q=blue",
    "/api/product/1001/reviews", "/api/recommendations",
]

BROWSER_HEADERS_FULL = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "upgrade-insecure-requests": "1",
}


# ---------------------------------------------------------------------------
# Request & Session dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Request:
    timestamp_ms: float           # Unix timestamp in milliseconds
    method: str                   # GET / POST / PUT
    url: str
    status_code: int
    response_time_ms: float       # server response latency
    headers: Dict[str, str]
    user_agent: str
    ip: str
    session_id: str
    request_size_bytes: int
    response_size_bytes: int
    is_api: bool = False
    referrer: Optional[str] = None


@dataclass
class Session:
    session_id: str
    ip: str
    user_agent: str
    traffic_class: TrafficClass
    requests: List[Request] = field(default_factory=list)
    start_time_ms: float = 0.0
    label: int = 0                # 0 = human, 1 = bot

    @property
    def n_requests(self) -> int:
        return len(self.requests)

    @property
    def duration_ms(self) -> float:
        if len(self.requests) < 2:
            return 0.0
        return self.requests[-1].timestamp_ms - self.requests[0].timestamp_ms

    @property
    def inter_arrival_times(self) -> List[float]:
        if len(self.requests) < 2:
            return []
        ts = [r.timestamp_ms for r in self.requests]
        return [ts[i+1] - ts[i] for i in range(len(ts)-1)]


# ---------------------------------------------------------------------------
# Traffic simulator
# ---------------------------------------------------------------------------

class TrafficSimulator:
    """
    Generates realistic synthetic HTTP sessions for all five traffic classes.

    Usage:
        sim = TrafficSimulator(cfg.simulation)
        sessions = sim.generate()
    """

    def __init__(self, sim_cfg, seed: int = 42):
        self.cfg = sim_cfg
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self._ip_counter = 0

    def generate(self) -> List[Session]:
        """Generate all sessions according to class fractions in config."""
        cfg = self.cfg
        n   = cfg.n_sessions

        counts = {
            TrafficClass.HUMAN:        int(n * cfg.human_fraction),
            TrafficClass.SIMPLE_BOT:   int(n * cfg.simple_bot_fraction),
            TrafficClass.SCRAPER:      int(n * cfg.scraper_fraction),
            TrafficClass.CRED_STUFFER: int(n * cfg.credential_stuffer_fraction),
            TrafficClass.LLM_AGENT:    int(n * cfg.llm_agent_fraction),
        }

        sessions: List[Session] = []
        for tc, count in counts.items():
            for _ in range(count):
                sessions.append(self._generate_session(tc))

        self.rng.shuffle(sessions)
        return sessions

    # ------------------------------------------------------------------
    # Session generators per class
    # ------------------------------------------------------------------

    def _generate_session(self, tc: TrafficClass) -> Session:
        dispatch = {
            TrafficClass.HUMAN:        self._human_session,
            TrafficClass.SIMPLE_BOT:   self._simple_bot_session,
            TrafficClass.SCRAPER:      self._scraper_session,
            TrafficClass.CRED_STUFFER: self._cred_stuffer_session,
            TrafficClass.LLM_AGENT:    self._llm_agent_session,
        }
        return dispatch[tc]()

    def _new_session(self, tc: TrafficClass, ua: str) -> Session:
        self._ip_counter += 1
        ip = f"192.168.{self._ip_counter // 256 % 256}.{self._ip_counter % 256}"
        sid = hashlib.md5(f"{ip}{tc}{self._ip_counter}".encode()).hexdigest()[:12]
        return Session(
            session_id=sid, ip=ip, user_agent=ua,
            traffic_class=tc,
            start_time_ms=float(self.rng.randint(1_700_000_000_000, 1_710_000_000_000)),
            label=0 if tc == TrafficClass.HUMAN else 1,
        )

    def _make_request(
        self, session: Session, t_ms: float, url: str,
        method: str = "GET", headers: Dict = None, is_api: bool = False,
        referrer: str = None,
    ) -> Request:
        status = 200
        # Simulate some 404s for bots crawling non-existent pages
        if url not in STORE_PAGES and session.traffic_class != TrafficClass.HUMAN:
            status = 404 if self.rng.random() < 0.15 else 200

        return Request(
            timestamp_ms=t_ms,
            method=method,
            url=url,
            status_code=status,
            response_time_ms=self.rng.gauss(80, 20),
            headers=headers or BROWSER_HEADERS_FULL,
            user_agent=session.user_agent,
            ip=session.ip,
            session_id=session.session_id,
            request_size_bytes=self.rng.randint(200, 2000),
            response_size_bytes=self.rng.randint(5000, 200000),
            is_api=is_api,
            referrer=referrer,
        )

    # ------------------------------------------------------------------
    # Human session
    # ------------------------------------------------------------------

    def _human_session(self) -> Session:
        """
        Humans browse with:
          - High IAT variance (reading, distracted, multitasking)
          - Non-linear navigation (back button, bookmarks)
          - Natural page depth (3–25 pages)
          - Full browser headers
        """
        ua = self.rng.choice(BROWSER_UAS)
        session = self._new_session(TrafficClass.HUMAN, ua)
        n_pages = self.rng.randint(*self.cfg.human_session_pages)
        t = session.start_time_ms

        pages = ["/"] + self.rng.sample(STORE_PAGES, min(n_pages, len(STORE_PAGES)))
        for i, page in enumerate(pages):
            # Human IAT: log-normal distribution (long tail)
            iat = max(500, np.random.lognormal(
                math.log(max(self.cfg.human_mean_iat_ms, 1)),
                0.8
            ))
            t += iat if i > 0 else 0
            ref = pages[i-1] if i > 0 else None
            session.requests.append(
                self._make_request(session, t, page, referrer=ref)
            )
            # Sometimes load sub-resources (JS, CSS, images)
            for _ in range(self.rng.randint(2, 8)):
                t += self.rng.uniform(10, 100)
                session.requests.append(
                    self._make_request(session, t, f"/static/{self.rng.randint(1,50)}.js")
                )

        return session

    # ------------------------------------------------------------------
    # Simple bot session
    # ------------------------------------------------------------------

    def _simple_bot_session(self) -> Session:
        """
        Naive scrapers:
          - Very regular timing (fixed sleep interval)
          - Missing or minimal headers
          - Bot user agent
          - Deep crawling
        """
        ua = self.rng.choice(BOT_UAS)
        session = self._new_session(TrafficClass.SIMPLE_BOT, ua)
        n_pages = self.rng.randint(*self.cfg.bot_session_pages)
        t = session.start_time_ms
        minimal_headers = {"accept": "*/*", "user-agent": ua}

        for i in range(n_pages):
            iat = self.rng.gauss(self.cfg.bot_mean_iat_ms, self.cfg.bot_stddev_iat_ms)
            t += max(50, iat) if i > 0 else 0
            url = self.rng.choice(STORE_PAGES)
            session.requests.append(
                self._make_request(session, t, url, headers=minimal_headers)
            )

        return session

    # ------------------------------------------------------------------
    # Sophisticated scraper session
    # ------------------------------------------------------------------

    def _scraper_session(self) -> Session:
        """
        Sophisticated scrapers:
          - Rotating user agents (appears as multiple browsers)
          - Moderate IAT with some variance (tries to blend in)
          - Targets specific data (products, prices)
          - Misses some browser headers (sec-ch-ua, sec-fetch-*)
        """
        ua = self.rng.choice(BROWSER_UAS)
        session = self._new_session(TrafficClass.SCRAPER, ua)
        n_pages = self.rng.randint(30, 200)
        t = session.start_time_ms
        # Missing sec-fetch headers — dead giveaway
        partial_headers = {
            "accept": "text/html,application/xhtml+xml,*/*",
            "accept-language": "en-US,en;q=0.9",
            "accept-encoding": "gzip, deflate",
            "user-agent": ua,
        }

        product_pages = [p for p in STORE_PAGES if "product" in p]
        for i in range(n_pages):
            # More regular than humans but some jitter
            iat = self.rng.gauss(1500, 400)
            t += max(100, iat) if i > 0 else 0
            url = self.rng.choice(product_pages + ["/products"])
            session.requests.append(
                self._make_request(session, t, url, headers=partial_headers)
            )

        return session

    # ------------------------------------------------------------------
    # Credential stuffer session
    # ------------------------------------------------------------------

    def _cred_stuffer_session(self) -> Session:
        """
        Credential stuffing:
          - Hits login endpoint repeatedly
          - High POST rate
          - Few unique pages (login → 401 → retry)
          - Often uses residential proxy IPs
        """
        ua = self.rng.choice(BROWSER_UAS)
        session = self._new_session(TrafficClass.CRED_STUFFER, ua)
        t = session.start_time_ms
        n_attempts = self.rng.randint(20, 200)

        for i in range(n_attempts):
            iat = self.rng.gauss(300, 50)
            t += max(100, iat) if i > 0 else 0
            session.requests.append(
                self._make_request(session, t, "/account/login", method="POST")
            )
            # Mostly get 401s
            session.requests[-1].status_code = (
                200 if self.rng.random() < 0.005 else 401
            )

        return session

    # ------------------------------------------------------------------
    # LLM-Agent session
    # ------------------------------------------------------------------

    def _llm_agent_session(self) -> Session:
        """
        LLM-powered agent (GPT-4o, Claude, Gemini with web tools):

        The LLM orchestrates browsing via tool calls. Each tool call
        introduces a processing delay that is surprisingly consistent
        (LLM inference latency is roughly constant for similar prompts).

        Key signatures:
          1. Regular IAT (LLM think → act cycle is metronomic)
          2. Systematic link following (BFS/DFS of the site graph)
          3. API probing (discovers /api/search, /api/autocomplete etc.)
          4. Deep sessions (100+ pages — LLM doesn't get tired or bored)
          5. Perfect referrer chain (visits every link on every page)
          6. Content-agnostic timing (same speed on 100-word vs 10,000-word pages)
        """
        ua = self.rng.choice(BROWSER_UAS)  # LLM agents often use headless Chrome UA
        session = self._new_session(TrafficClass.LLM_AGENT, ua)
        n_pages = self.rng.randint(*self.cfg.llm_session_pages)
        t = session.start_time_ms

        # LLM agents tend to start at the homepage and go breadth-first
        queue = list(STORE_PAGES)
        self.rng.shuffle(queue)
        visited = []

        headers = {**BROWSER_HEADERS_FULL}  # LLM agents using Playwright have full headers

        for i in range(min(n_pages, len(queue))):
            url = queue[i]
            visited.append(url)

            # Signature 1: Very regular IAT — LLM inference latency is consistent
            base_iat = self.cfg.llm_mean_iat_ms
            iat = self.rng.gauss(base_iat, self.cfg.llm_stddev_iat_ms)
            iat = max(300, iat)
            t += iat if i > 0 else 0

            is_api = "/api/" in url
            ref = visited[-2] if len(visited) > 1 else None
            session.requests.append(
                self._make_request(session, t, url, headers=headers,
                                   is_api=is_api, referrer=ref)
            )

            # Signature 2: LLM agent probes APIs it discovers
            if not is_api and self.rng.random() < 0.3:
                t += self.rng.gauss(600, 80)
                api_url = self.rng.choice([
                    "/api/search?q=product",
                    "/api/autocomplete?q=blue",
                    "/api/recommendations",
                    "/api/product/1001/reviews",
                ])
                session.requests.append(
                    self._make_request(session, t, api_url, is_api=True,
                                       headers=headers, referrer=url)
                )

        return session


# ---------------------------------------------------------------------------
# Utility: sessions → flat request log
# ---------------------------------------------------------------------------

def sessions_to_log(sessions: List[Session]) -> List[Dict]:
    """Flatten all sessions into a list of request-level dicts."""
    rows = []
    for s in sessions:
        for r in s.requests:
            rows.append({
                "session_id":        s.session_id,
                "traffic_class":     s.traffic_class.value,
                "label":             s.label,
                "ip":                s.ip,
                "timestamp_ms":      r.timestamp_ms,
                "method":            r.method,
                "url":               r.url,
                "status_code":       r.status_code,
                "response_time_ms":  r.response_time_ms,
                "user_agent":        r.user_agent,
                "is_api":            r.is_api,
                "request_size_bytes": r.request_size_bytes,
                "response_size_bytes": r.response_size_bytes,
                "n_headers":         len(r.headers),
                "has_referrer":      r.referrer is not None,
            })
    return rows
