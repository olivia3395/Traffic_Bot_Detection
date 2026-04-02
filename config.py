"""
config.py — Central configuration for the Bot Detection & Mitigation System.

System Overview
───────────────
This system protects e-commerce properties from automated threats by
combining three complementary detection layers:

  Layer 1 — Statistical detectors
      Fast, rule-based, near-zero latency.
      Catches high-volume, unsophisticated bots immediately.

  Layer 2 — ML-based anomaly & classification detectors
      Isolation Forest (unsupervised) + Gradient Boosting (supervised).
      Catches medium-sophistication bots that evade simple rules.

  Layer 3 — LLM-agent fingerprinting
      Detects the unique behavioral signatures of LLM-powered agents
      (GPT, Claude, Gemini, etc.) orchestrating automated browsing.
      These are the most sophisticated threats and require specialised
      signal extraction.

  Ensemble — Weighted combination of all layers → final risk score [0,1]
  Mitigation — Route decision (allow / challenge / throttle / block)
               based on risk score and business context.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature extraction configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Controls which signals are extracted from traffic."""

    # Session window for aggregating features
    session_window_seconds: int = 300      # 5-minute rolling window

    # Minimum requests before scoring a session
    min_requests_to_score: int = 3

    # Inter-arrival time (IAT) — time between consecutive requests
    iat_bins: int = 20                     # bins for IAT histogram

    # URL entropy — how diverse is the page browsing?
    url_entropy_window: int = 50           # last N URLs

    # LLM-specific: how many consecutive "perfect" patterns trigger suspicion
    llm_regularity_threshold: int = 5

    # TLS / JA3 fingerprint known-bot list (hash → label)
    known_bot_ja3: List[str] = field(default_factory=lambda: [
        "e7d705a3286e19ea42f587b344ee6865",   # Scrapy
        "5765db2da7f2bfc028a93aca42df2e45",   # Python requests
        "b32309a26951912be7dba376398abc3b",   # curl/wget
        "3b5074b1b5d032e5620f69f9f700ff0e",   # Selenium default
    ])

    # HTTP headers that real browsers always send
    required_browser_headers: List[str] = field(default_factory=lambda: [
        "accept",
        "accept-language",
        "accept-encoding",
        "sec-fetch-dest",
        "sec-ch-ua",
    ])


# ---------------------------------------------------------------------------
# Detector configuration
# ---------------------------------------------------------------------------

@dataclass
class StatisticalDetectorConfig:
    """Rule-based and statistical anomaly thresholds."""

    # Request rate thresholds (requests per minute)
    rate_warn_rpm: float  = 60.0           # Yellow flag
    rate_block_rpm: float = 300.0          # Hard block

    # IAT (inter-arrival time) regularity
    # Real humans have high coefficient of variation (CV > 1.5)
    # Bots tend to be regular (CV < 0.3 for naive bots)
    iat_cv_bot_threshold: float = 0.4      # CV below this → suspicious

    # Z-score threshold for request rate anomaly
    zscore_threshold: float = 3.0

    # Consecutive 4xx errors (crawlers hit non-existent pages)
    max_consecutive_404s: int = 5

    # Session depth (pages per session — very deep → scraper)
    max_human_depth: int = 50


@dataclass
class MLDetectorConfig:
    """Machine learning detector settings."""

    # Isolation Forest
    if_contamination: float  = 0.05       # expected fraction of anomalies
    if_n_estimators: int     = 100
    if_max_samples: str      = "auto"
    if_random_state: int     = 42

    # Gradient Boosting classifier
    gb_n_estimators: int     = 200
    gb_max_depth: int        = 5
    gb_learning_rate: float  = 0.05
    gb_random_state: int     = 42
    gb_subsample: float      = 0.8

    # Minimum training samples needed before ML model is trusted
    min_training_samples: int = 500

    # Feature selection: top-k features by importance
    top_k_features: int = 30


@dataclass
class LLMDetectorConfig:
    """
    Configuration for LLM-agent specific detection.

    LLM-powered agents exhibit distinctive patterns:
      1. Hyper-regularity — LLM planning produces unnaturally uniform timing
      2. Systematic coverage — methodically visits all links, no backtracking
      3. Content-agnostic depth — reads product pages same speed as homepages
      4. Structured API probing — discovers and calls internal APIs systematically
      5. Natural language coherence — form inputs are grammatically perfect
      6. Zero hesitation — no mouse dwell time, no re-reads
      7. Context window exhaustion — very long sessions with linear progression
    """

    # Timing regularity: stddev of IAT below this → LLM-like
    timing_stddev_threshold_ms: float = 150.0

    # Link coverage ratio: if agent visits >80% of discovered links → systematic
    link_coverage_threshold: float = 0.80

    # Semantic coherence of visited URLs (LLM agents follow logical paths)
    url_coherence_threshold: float = 0.75

    # Suspicion score weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "timing_regularity":    0.20,
        "systematic_coverage":  0.20,
        "header_anomaly":       0.15,
        "ua_mismatch":          0.10,
        "api_probing":          0.15,
        "form_naturalness":     0.10,
        "session_linearity":    0.10,
    })


@dataclass
class EnsembleConfig:
    """Ensemble combiner settings."""

    # Weights for each detector layer
    weights: Dict[str, float] = field(default_factory=lambda: {
        "statistical": 0.25,
        "isolation_forest": 0.25,
        "gradient_boosting": 0.30,
        "llm_detector": 0.20,
    })

    # If any single detector is this confident, override ensemble
    hard_override_threshold: float = 0.95

    # Minimum detectors that must agree before high-confidence block
    min_agreement: int = 2


# ---------------------------------------------------------------------------
# Mitigation configuration
# ---------------------------------------------------------------------------

@dataclass
class MitigationConfig:
    """
    Risk score → action mapping.

    Score ranges:
      0.0 – 0.30 : ALLOW      — normal traffic
      0.30 – 0.55: MONITOR    — log and watch
      0.55 – 0.70: THROTTLE   — slow down response (rate limit)
      0.70 – 0.85: CHALLENGE  — CAPTCHA / JS challenge
      0.85 – 1.00: BLOCK      — hard block
    """

    thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "allow":     (0.00, 0.30),
        "monitor":   (0.30, 0.55),
        "throttle":  (0.55, 0.70),
        "challenge": (0.70, 0.85),
        "block":     (0.85, 1.00),
    })

    # Throttle delay in seconds
    throttle_delay_seconds: float = 2.0

    # Block duration
    block_duration_seconds: int = 3600      # 1 hour

    # Challenge type ("captcha" | "js_proof_of_work" | "email_verify")
    challenge_type: str = "js_proof_of_work"

    # Allowlist: IPs / ASNs that bypass detection (known good crawlers)
    allowlist_asns: List[str] = field(default_factory=lambda: [
        "AS15169",   # Google
        "AS8075",    # Microsoft Bing
        "AS16509",   # Amazon
    ])



# ---------------------------------------------------------------------------
# CSIC 2010 dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class CSICConfig:
    """
    Configuration for the CSIC 2010 HTTP Intrusion Detection Dataset.

    Download from one of:
        https://www.kaggle.com/datasets/victorsolano/http-dataset-csic-2010
        http://www.isi.csic.es/dataset/

    Place the three .txt files in data_dir:
        normalTrafficTraining.txt
        normalTrafficTest.txt
        anomalousTrafficTest.txt
    """
    data_dir: str               = "data/csic2010"
    normal_train_file: str      = "normalTrafficTraining.txt"
    normal_test_file: str       = "normalTrafficTest.txt"
    anomalous_file: str         = "anomalousTrafficTest.txt"
    session_size: int           = 20
    max_normal_sessions: Optional[int]    = None
    max_anomalous_sessions: Optional[int] = None
    add_synthetic_llm_agents: bool = True
    n_synthetic_llm_sessions: int  = 200


# ---------------------------------------------------------------------------
# Simulation configuration (for generating synthetic traffic data)
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Settings for generating synthetic traffic datasets."""

    random_seed: int = 42

    # Total sessions to generate
    n_sessions: int = 5000

    # Class distribution
    human_fraction: float        = 0.70
    simple_bot_fraction: float   = 0.10
    scraper_fraction: float      = 0.08
    credential_stuffer_fraction: float = 0.05
    llm_agent_fraction: float    = 0.07   # LLM-powered agents

    # Human behaviour parameters
    human_mean_iat_ms: float   = 4500.0   # ~4.5s between clicks
    human_stddev_iat_ms: float = 3200.0   # high variance (distracted, reading)
    human_session_pages: Tuple[int, int] = (3, 25)

    # LLM agent behaviour parameters
    llm_mean_iat_ms: float   = 800.0     # fast, but not instant
    llm_stddev_iat_ms: float = 120.0     # very regular
    llm_session_pages: Tuple[int, int] = (20, 150)

    # Simple bot parameters
    bot_mean_iat_ms: float   = 200.0
    bot_stddev_iat_ms: float = 30.0
    bot_session_pages: Tuple[int, int] = (50, 500)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    features:    FeatureConfig           = field(default_factory=FeatureConfig)
    statistical: StatisticalDetectorConfig = field(default_factory=StatisticalDetectorConfig)
    ml:          MLDetectorConfig        = field(default_factory=MLDetectorConfig)
    llm:         LLMDetectorConfig       = field(default_factory=LLMDetectorConfig)
    ensemble:    EnsembleConfig          = field(default_factory=EnsembleConfig)
    mitigation:  MitigationConfig        = field(default_factory=MitigationConfig)
    simulation:  SimulationConfig        = field(default_factory=SimulationConfig)
    csic:        CSICConfig              = field(default_factory=CSICConfig)

    # Output
    output_dir: str = "outputs"
    log_level: str  = "INFO"

    def summary(self) -> str:
        lines = ["=" * 60, "  Bot Detection System — Configuration", "=" * 60]
        for sec in ("features", "statistical", "ml", "llm", "ensemble", "mitigation"):
            obj = getattr(self, sec)
            lines.append(f"\n[{sec.upper()}]")
            for k, v in vars(obj).items():
                lines.append(f"  {k:<35} {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
