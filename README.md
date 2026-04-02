# Bot Detection & Mitigation System
### Next-Generation Automated Threat Protection


Modern bot threats range from naive scrapers to LLM-orchestrated agents (GPT-4o, Claude, Gemini with web browsing tools) that mimic human browsing at a high level. This system detects all threat classes through a **three-layer ensemble**:

| Layer | Detector | Latency | Catches |
|-------|----------|---------|---------|
| 1 | Statistical rules | < 1ms | Naive bots, obvious anomalies |
| 2a | Isolation Forest | < 5ms | Novel patterns, unsupervised |
| 2b | Gradient Boosting | < 5ms | All bot classes, high accuracy |
| 3 | LLM Fingerprinting | < 5ms | LLM-powered agents specifically |
| ∑ | **Ensemble** | **< 15ms** | **All threats** |


## Threat Classes

| Class | Description | Key Signals |
|-------|-------------|-------------|
| `human` | Organic user browsing | High IAT variance, non-linear navigation, backtracking |
| `simple_bot` | Naive scrapers | Bot UA, missing headers, very regular timing |
| `scraper` | Sophisticated crawlers | Partial headers, product-focused, moderate regularity |
| `cred_stuffer` | Credential stuffing | High POST rate, repeated login attempts, few pages |
| `llm_agent` | **LLM-powered agents** | **Moderate+regular timing, systematic coverage, API probing** |



## LLM-Agent Detection

The newest and most sophisticated threat. LLM agents (GPT-4o with browsing, Claude computer use, Gemini web agents) produce a distinctive "LLM heartbeat":

```
User prompt
    ↓
LLM reasons about next action (700-1500ms, very consistent)
    ↓
Issues browser tool call (GET /products/shoes)
    ↓
Receives page content
    ↓
LLM reasons about next action ...
```

**Seven fingerprint signals:**

| Signal | Description | Human | LLM Agent |
|--------|-------------|-------|-----------|
| Timing regularity | IAT coefficient of variation | CV > 0.8 | CV ≈ 0.1–0.2 |
| Systematic coverage | Visits all discovered links | ~40% | ~80–95% |
| Header anomaly | Missing/inconsistent headers | Complete | Near-complete |
| UA consistency | Same UA throughout session | Variable | Consistent |
| API probing | Discovers & calls internal APIs | Rare | Systematic |
| Form naturalness | Perfect grammar, no typos | Rare | Always |
| Session linearity | Forward-only navigation | Low | High |



### Install
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



## Performance

On 5,000 synthetic sessions at FPR ≤ 1%:

| Metric | Target | Typical Result |
|--------|--------|---------------|
| Overall Recall | > 90% | ~93–96% |
| FPR (false block rate) | < 1% | ~0.4–0.8% |
| LLM Agent Recall | > 85% | ~88–92% |
| Simple Bot Recall | > 99% | ~99–100% |
| AUROC | > 0.97 | ~0.98–0.99 |
| AUPRC | > 0.95 | ~0.96–0.98 |



## Feature Importance (Top 10)

From Gradient Boosting on 5,000 sessions:

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



## Mitigation Actions

```
Risk Score     Action      Description
──────────────────────────────────────────────────────
0.00 – 0.30   ALLOW       Serve normally, log for audit
0.30 – 0.55   MONITOR     Serve but flag for analyst review
0.55 – 0.70   THROTTLE    Add 2s delay, limit to 10 rpm
0.70 – 0.85   CHALLENGE   JS proof-of-work / CAPTCHA
0.85 – 1.00   BLOCK       HTTP 403, log IP+fingerprint, alert
```

Context-aware adjustment: login/checkout pages apply a -0.10 threshold shift (more aggressive protection for high-value endpoints).



## Configuration

All thresholds, weights, and parameters are in `config.py`:

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


