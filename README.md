<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Latency-<15ms-22c55e?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AUROC->0.98-6366f1?style=for-the-badge"/>

<br/><br/>

# 🛡️ Bot Detection & Mitigation System
### Next-Generation Automated Threat Protection

<br/>

> Detects everything from naive scrapers to **LLM-orchestrated agents**  
> (GPT-4o, Claude, Gemini) through a real-time three-layer ensemble — in **< 15ms**.

<br/>

[🚀 Quick Start](#-quick-start) · [🔍 Threat Classes](#-threat-classes) · [🤖 LLM-Agent Detection](#-llm-agent-detection) · [📊 Performance](#-performance) · [⚙️ Configuration](#️-configuration)

<br/>



</div>

## 🏗️ Three-Layer Ensemble

Requests pass through three detection layers in parallel, each targeting a different slice of the threat landscape:

<br/>

```
Incoming Request
       │
       ├──► Layer 1 · Statistical Rules    < 1ms   Naive bots, obvious anomalies
       │
       ├──► Layer 2a · Isolation Forest    < 5ms   Novel patterns (unsupervised)
       ├──► Layer 2b · Gradient Boosting   < 5ms   All bot classes, high accuracy
       │
       └──► Layer 3 · LLM Fingerprinting   < 5ms   LLM-powered agents specifically
                             │
                       Ensemble Vote
                             │
                      Risk Score [0–1]
                             │
                      Mitigation Action
```

<br/>

<div align="center">

| Layer | Detector | Latency | Catches |
|:---:|:---|:---:|:---|
| 1 | Statistical Rules | `< 1ms` | Naive bots, obvious anomalies |
| 2a | Isolation Forest | `< 5ms` | Novel patterns, unsupervised |
| 2b | Gradient Boosting | `< 5ms` | All bot classes, high accuracy |
| 3 | LLM Fingerprinting | `< 5ms` | LLM-powered agents specifically |
| **∑** | **Ensemble** | **`< 15ms`** | **All threat classes** |

</div>

<br/>



## 🔍 Threat Classes

<div align="center">

| Class | Description | Key Signals |
|:---:|:---|:---|
| 👤 `human` | Organic user browsing | High IAT variance · non-linear navigation · backtracking |
| 🤖 `simple_bot` | Naive scrapers | Bot UA · missing headers · very regular timing |
| 🕷️ `scraper` | Sophisticated crawlers | Partial headers · product-focused · moderate regularity |
| 🔑 `cred_stuffer` | Credential stuffing | High POST rate · repeated login attempts · few pages |
| 🧠 `llm_agent` | **LLM-powered agents** | **Consistent timing · systematic coverage · API probing** |

</div>

<br/>



## 🤖 LLM-Agent Detection

The newest and most sophisticated threat class. LLM agents (GPT-4o with browsing, Claude computer use, Gemini web agents) produce a distinctive **"LLM heartbeat"** pattern:

```
User prompt
    │
    ▼
LLM reasons about next action        ← 700–1500ms, highly consistent
    │
    ▼
Issues browser tool call             ← GET /products/shoes
    │
    ▼
Receives page content
    │
    ▼
LLM reasons about next action ...    ← cycle repeats with near-identical intervals
```

<br/>

### 7 Fingerprint Signals

<div align="center">

| Signal | Description | Human | LLM Agent |
|:---|:---|:---:|:---:|
| ⏱️ **Timing regularity** | IAT coefficient of variation | CV > 0.8 | CV ≈ 0.1–0.2 |
| 🗺️ **Systematic coverage** | Fraction of discovered links visited | ~40% | ~80–95% |
| 📋 **Header anomaly** | Missing or inconsistent HTTP headers | Complete | Near-complete |
| 🪪 **UA consistency** | Same User-Agent throughout session | Variable | Consistent |
| 🔌 **API probing** | Discovers & calls internal API endpoints | Rare | Systematic |
| ✍️ **Form naturalness** | Perfect grammar, zero typos in form fields | Rare | Always |
| ➡️ **Session linearity** | Forward-only navigation (no backtracking) | Low | High |

</div>

<br/>



## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train detectors

```bash
python scripts/train.py --n-sessions 5000
```

### Run live demo

```bash
python scripts/demo.py
```

### Evaluate on fresh traffic

```bash
python scripts/evaluate.py --n-sessions 2000 --target-fpr 0.01
```

### Run unit tests

```bash
python tests/test_all.py
```

<br/>



## 📊 Performance

Evaluated on **5,000 synthetic sessions** at FPR ≤ 1%:

<div align="center">

| Metric | Target | Typical Result |
|:---|:---:|:---:|
| Overall Recall | > 90% | **~93–96%** |
| False Positive Rate | < 1% | **~0.4–0.8%** |
| LLM Agent Recall | > 85% | **~88–92%** |
| Simple Bot Recall | > 99% | **~99–100%** |
| AUROC | > 0.97 | **~0.98–0.99** |
| AUPRC | > 0.95 | **~0.96–0.98** |

</div>

<br/>

### 🔬 Feature Importance (Top 10)

From Gradient Boosting trained on 5,000 sessions:

```
  llm_timing_regularity          ████████████  0.089
  http_iat_cv                    ███████████   0.081
  beh_dwell_cv                   ██████████    0.074
  http_header_completeness       █████████     0.068
  llm_systematic_coverage        █████████     0.063
  beh_session_linearity          ████████      0.058
  http_rpm_mean                  ████████      0.055
  llm_api_probing                ███████       0.049
  beh_backtrack_rate             ███████       0.047
  http_burst_ratio               ██████        0.041
```

<br/>


## 🚦 Mitigation Actions

Risk scores are mapped to five progressive response tiers:

```
Risk Score      Action       Description
─────────────────────────────────────────────────────────────────
0.00 – 0.30    ✅ ALLOW      Serve normally, log for audit
0.30 – 0.55    👁️ MONITOR    Serve but flag for analyst review
0.55 – 0.70    🐢 THROTTLE   Add 2s delay, limit to 10 rpm
0.70 – 0.85    🧩 CHALLENGE  JS proof-of-work / CAPTCHA
0.85 – 1.00    🚫 BLOCK      HTTP 403, log IP + fingerprint, alert
```

> ⚠️ **Context-aware adjustment:** Login and checkout endpoints apply a `−0.10` threshold shift for more aggressive protection of high-value pages.

<br/>



## ⚙️ Configuration

All thresholds, weights, and parameters live in `config.py`:

```python
# Adjust ensemble weights per environment
cfg.ensemble.weights = {
    "statistical":       0.25,
    "isolation_forest":  0.25,
    "gradient_boosting": 0.30,
    "llm_detector":      0.20,
}

# Tune for lower FPR (safer) or higher recall (more aggressive)
cfg.mitigation.thresholds["block"] = (0.90, 1.00)   # more conservative
```

<br/>



<div align="center">



</div>
