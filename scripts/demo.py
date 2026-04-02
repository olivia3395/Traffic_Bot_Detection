"""
scripts/demo.py — Live bot detection demo.

Generates one session per traffic class and shows the full
detection pipeline output: per-layer scores, ensemble decision,
and mitigation action.

Usage:
  python scripts/demo.py
  python scripts/demo.py --n-per-class 3
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data.simulator import TrafficSimulator, TrafficClass
from detectors.statistical import StatisticalDetector
from detectors.ml_detector import IsolationForestDetector, GradientBoostingDetector
from detectors.llm_detector import LLMAgentDetector
from detectors.ensemble import EnsembleDetector
from mitigation.strategies import MitigationEngine
from features.feature_pipeline import build_dataset


COLORS = {
    "allow":     "\033[32m",   # green
    "monitor":   "\033[36m",   # cyan
    "throttle":  "\033[33m",   # yellow
    "challenge": "\033[35m",   # magenta
    "block":     "\033[31m",   # red
    "reset":     "\033[0m",
    "bold":      "\033[1m",
    "dim":       "\033[2m",
}


def parse_args():
    p = argparse.ArgumentParser(description="Bot detection live demo")
    p.add_argument("--n-per-class", type=int, default=2)
    p.add_argument("--model-dir",   default="outputs")
    p.add_argument("--no-color",    action="store_true")
    return p.parse_args()


def colorize(text, color_key, args):
    if args.no_color:
        return text
    return f"{COLORS.get(color_key,'')}{text}{COLORS['reset']}"


def main():
    args = parse_args()
    cfg  = Config()

    print(f"\n{colorize('='*62, 'bold', args)}")
    print(f"  Bot Detection & Mitigation — Live Demo")
    print(f"  Team: Stores-Traffic Engineering")
    print(f"{'='*62}\n")

    # Build detectors
    print("Loading detectors...")
    stat_det = StatisticalDetector(cfg)
    llm_det  = LLMAgentDetector(cfg)

    import os
    if_path = os.path.join(args.model_dir, "isolation_forest.pkl")
    gb_path = os.path.join(args.model_dir, "gradient_boosting.pkl")

    if os.path.exists(if_path) and os.path.exists(gb_path):
        if_det = IsolationForestDetector.load(if_path, cfg)
        gb_det = GradientBoostingDetector.load(gb_path, cfg)
    else:
        print("  No saved models — training a quick model for demo...")
        sim      = TrafficSimulator(cfg.simulation, seed=42)
        sessions = sim.generate()
        dataset  = build_dataset(sessions, cfg)
        if_det   = IsolationForestDetector(cfg)
        if_det.fit(dataset)
        gb_det   = GradientBoostingDetector(cfg)
        gb_det.fit(dataset)

    ensemble   = EnsembleDetector(cfg, if_det, gb_det, llm_det, stat_det)
    mitigation = MitigationEngine(cfg)

    # Generate one session per class
    cfg.simulation.n_sessions = args.n_per_class * 5
    sim = TrafficSimulator(cfg.simulation, seed=999)
    all_sessions = sim.generate()

    class_sessions = {}
    for s in all_sessions:
        tc = s.traffic_class
        if tc not in class_sessions:
            class_sessions[tc] = []
        if len(class_sessions[tc]) < args.n_per_class:
            class_sessions[tc].append(s)

    ordered = [
        TrafficClass.HUMAN,
        TrafficClass.SIMPLE_BOT,
        TrafficClass.SCRAPER,
        TrafficClass.CRED_STUFFER,
        TrafficClass.LLM_AGENT,
    ]

    for tc in ordered:
        sessions = class_sessions.get(tc, [])
        for session in sessions:
            print(f"\n{'─'*62}")
            tc_label = session.traffic_class.value.upper().replace("_", " ")
            print(f"  Session: {colorize(tc_label, 'bold', args)}"
                  f"  |  ID={session.session_id}"
                  f"  |  {len(session.requests)} requests")

            # Score
            result   = ensemble.score_session(session)
            decision = mitigation.decide(result, session)

            # Display per-layer scores
            print(f"\n  Detection Layers:")
            layers = [
                ("Statistical",      result["statistical_score"]),
                ("Isolation Forest", result["isolation_score"]),
                ("Gradient Boosting",result["gb_score"]),
                ("LLM Fingerprint",  result["llm_score"]),
            ]
            for name, score in layers:
                bar_len = int(score * 30)
                bar     = "█" * bar_len + "░" * (30 - bar_len)
                flag    = " ⚠" if score > 0.5 else ""
                print(f"    {name:<22} [{bar}] {score:.3f}{flag}")

            # Ensemble result
            risk = result["risk_score"]
            action_color = decision.action.value
            print(f"\n  {'─'*40}")
            print(f"  Ensemble Risk Score : {colorize(f'{risk:.3f}', action_color, args)}")
            print(f"  Detectors Firing    : {result['n_detectors_firing']}/4")

            # Mitigation action
            action_str = decision.action.value.upper()
            print(f"  Mitigation Action   : {colorize(action_str, action_color, args)}", end="")
            if decision.delay_ms:
                print(f"  (delay={decision.delay_ms}ms)", end="")
            if decision.block_ttl_s:
                print(f"  (block {decision.block_ttl_s}s)", end="")
            if decision.challenge_type:
                print(f"  (challenge={decision.challenge_type})", end="")
            print()

            # LLM-specific signals for LLM agent sessions
            if tc == TrafficClass.LLM_AGENT and result.get("llm_signals"):
                print(f"\n  LLM Fingerprint Signals:")
                for sig, val in result["llm_signals"].items():
                    if sig == "llm_score":
                        continue
                    indicator = colorize("⚠", "block", args) if val > 0.5 else colorize("✓", "allow", args)
                    print(f"    {indicator} {sig:<25} {val:.3f}")

    print(f"\n{'═'*62}")
    print("  Demo complete.")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
