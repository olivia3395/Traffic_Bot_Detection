"""
scripts/evaluate.py — Evaluate trained detectors on test traffic.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --n-sessions 2000 --target-fpr 0.005
"""

import argparse, os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import Config
from data.simulator import TrafficSimulator
from features.feature_pipeline import build_dataset
from detectors.statistical import StatisticalDetector
from detectors.ml_detector import IsolationForestDetector, GradientBoostingDetector
from detectors.llm_detector import LLMAgentDetector
from detectors.ensemble import EnsembleDetector
from evaluation.metrics import (
    compute_metrics, per_class_metrics,
    find_optimal_threshold, print_eval_report,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sessions",  type=int,   default=2000)
    p.add_argument("--model-dir",   default="outputs")
    p.add_argument("--target-fpr",  type=float, default=0.01)
    p.add_argument("--seed",        type=int,   default=123)   # different from training seed
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()
    cfg.simulation.n_sessions  = args.n_sessions
    cfg.simulation.random_seed = args.seed

    print(f"\n{'═'*60}")
    print("  Bot Detection System — Evaluation")
    print(f"{'═'*60}")

    # Generate fresh traffic (different seed from training)
    print(f"\n[1/3] Generating {args.n_sessions} test sessions (seed={args.seed})...")
    sim      = TrafficSimulator(cfg.simulation, seed=args.seed)
    sessions = sim.generate()

    # Load trained ML models if available, else train on the fly
    print("\n[2/3] Loading detectors...")
    stat_det = StatisticalDetector(cfg)
    llm_det  = LLMAgentDetector(cfg)

    if_path = os.path.join(args.model_dir, "isolation_forest.pkl")
    gb_path = os.path.join(args.model_dir, "gradient_boosting.pkl")

    if os.path.exists(if_path) and os.path.exists(gb_path):
        if_det = IsolationForestDetector.load(if_path, cfg)
        gb_det = GradientBoostingDetector.load(gb_path, cfg)
        print(f"  Loaded models from {args.model_dir}")
    else:
        print("  No saved models found — training on the fly...")
        dataset = build_dataset(sessions[:int(len(sessions)*0.6)], cfg)
        if_det  = IsolationForestDetector(cfg)
        if_det.fit(dataset)
        gb_det  = GradientBoostingDetector(cfg)
        gb_det.fit(dataset)
        sessions = sessions[int(len(sessions)*0.6):]

    ensemble = EnsembleDetector(cfg, if_det, gb_det, llm_det, stat_det)

    # Score all sessions
    print(f"\n[3/3] Scoring {len(sessions)} sessions...")
    results      = ensemble.score_batch(sessions)
    y_scores     = np.array([r["risk_score"] for r in results])
    y_true       = np.array([s.label for s in sessions])
    test_classes = [s.traffic_class.value for s in sessions]

    # Per-detector breakdown
    print("\n  Per-Detector Scores (mean by class):")
    from collections import defaultdict
    class_scores = defaultdict(lambda: defaultdict(list))
    for s, r in zip(sessions, results):
        cls = s.traffic_class.value
        for key in ("statistical_score","isolation_score","gb_score","llm_score","risk_score"):
            class_scores[cls][key].append(r[key])

    headers = ["Class", "Stat", "IF", "GB", "LLM", "Ensemble"]
    print(f"  {'Class':<16} {'Stat':>7} {'IF':>7} {'GB':>7} {'LLM':>7} {'Ensemble':>9}")
    print(f"  {'─'*56}")
    for cls in sorted(class_scores):
        d = class_scores[cls]
        row = [
            np.mean(d["statistical_score"]),
            np.mean(d["isolation_score"]),
            np.mean(d["gb_score"]),
            np.mean(d["llm_score"]),
            np.mean(d["risk_score"]),
        ]
        print(f"  {cls:<16} " + " ".join(f"{v:>7.3f}" for v in row))

    # Optimal threshold evaluation
    opt_thr, opt_metrics = find_optimal_threshold(y_true, y_scores, target_fpr=args.target_fpr)
    per_cls = per_class_metrics(test_classes, y_true, y_scores, threshold=opt_thr)
    print_eval_report(opt_metrics, per_cls, opt_thr, label=f"Ensemble (target FPR≤{args.target_fpr})")

    # Save
    out = {
        "n_sessions": len(sessions),
        "seed": args.seed,
        "target_fpr": args.target_fpr,
        "optimal_threshold": opt_thr,
        "metrics": opt_metrics,
        "per_class": per_cls,
    }
    path = os.path.join(args.model_dir, "eval_results.json")
    os.makedirs(args.model_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved → {path}")


if __name__ == "__main__":
    main()
