"""
scripts/train.py — Train all bot detection models.

Supports two data sources:
  --data-source synthetic   Generate synthetic traffic (default, no download)
  --data-source csic        Load real CSIC 2010 HTTP dataset

CSIC 2010 download:
  https://www.kaggle.com/datasets/victorsolano/http-dataset-csic-2010
  Place the three .txt files in:  data/csic2010/

Usage:
  # Synthetic (offline, no download)
  python scripts/train.py

  # Real CSIC 2010 data (download first)
  python scripts/train.py --data-source csic --data-dir data/csic2010

  # CSIC + synthetic LLM agents (hybrid — recommended)
  python scripts/train.py --data-source csic --add-llm-agents
"""

import argparse, os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import Config
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
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-source",   default="synthetic", choices=["synthetic","csic"])
    p.add_argument("--data-dir",      default="data/csic2010")
    p.add_argument("--add-llm-agents",action="store_true", default=True)
    p.add_argument("--n-sessions",    type=int,   default=5000)
    p.add_argument("--session-size",  type=int,   default=20)
    p.add_argument("--output-dir",    default="outputs")
    p.add_argument("--target-fpr",    type=float, default=0.01)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def load_synthetic(cfg, args):
    from data.simulator import TrafficSimulator
    print(f"\n[Data] Generating {args.n_sessions} synthetic sessions...")
    return TrafficSimulator(cfg.simulation, seed=args.seed).generate()


def load_csic(cfg, args):
    from data.csic_loader import load_csic_dataset, attack_type_breakdown

    print(f"\n[Data] Loading CSIC 2010 from: {args.data_dir}")
    sessions = load_csic_dataset(
        data_dir=args.data_dir,
        session_size=args.session_size,
        seed=args.seed,
    )

    # Attack type breakdown
    print("\n  Attack type breakdown:")
    try:
        breakdown = attack_type_breakdown(args.data_dir)
        for atype, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"    {atype:<22} {count:>6,}  {'█'*min(count//20,40)}")
    except Exception as e:
        print(f"  (unavailable: {e})")

    # Hybrid: add synthetic LLM-agent sessions (CSIC 2010 has none — it's from 2010)
    if args.add_llm_agents:
        n_llm = cfg.csic.n_synthetic_llm_sessions
        print(f"\n  Adding {n_llm} synthetic LLM-agent sessions (hybrid mode)...")
        from data.simulator import TrafficSimulator
        cfg.simulation.n_sessions                    = n_llm
        cfg.simulation.human_fraction                = 0.0
        cfg.simulation.simple_bot_fraction           = 0.0
        cfg.simulation.scraper_fraction              = 0.0
        cfg.simulation.credential_stuffer_fraction   = 0.0
        cfg.simulation.llm_agent_fraction            = 1.0
        llm_sessions = TrafficSimulator(cfg.simulation, seed=args.seed+100).generate()
        sessions     = sessions + llm_sessions
        print(f"  Total after hybrid merge: {len(sessions):,}")

    return sessions


def main():
    args = parse_args()
    cfg  = Config()
    cfg.simulation.n_sessions  = args.n_sessions
    cfg.simulation.random_seed = args.seed
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Bot Detection — Training  |  data={args.data_source.upper()}")
    print(f"{'='*60}")

    # 1. Load data
    sessions = load_csic(cfg, args) if args.data_source == "csic" else load_synthetic(cfg, args)

    # 2. Feature extraction
    print("\n[2/5] Extracting features...")
    dataset  = build_dataset(sessions, cfg)
    train_ds, test_ds = dataset.train_test_split(test_fraction=0.2, seed=args.seed)
    print(f"  Train={len(train_ds)}  Test={len(test_ds)}")
    print(f"  Train distribution: {train_ds.class_distribution()}")

    # 3. Isolation Forest
    print("\n[3/5] Training Isolation Forest...")
    if_det = IsolationForestDetector(cfg); if_det.fit(train_ds)
    if_det.save(os.path.join(args.output_dir, "isolation_forest.pkl"))

    # 4. Gradient Boosting
    print("\n[4/5] Training Gradient Boosting...")
    gb_det = GradientBoostingDetector(cfg); gb_det.fit(train_ds)
    gb_det.save(os.path.join(args.output_dir, "gradient_boosting.pkl"))

    # 5. Ensemble evaluation
    print("\n[5/5] Evaluating ensemble on test set...")
    ensemble      = EnsembleDetector(cfg, if_det, gb_det, LLMAgentDetector(cfg), StatisticalDetector(cfg))
    test_sessions = sessions[-len(test_ds):]
    scores_list   = ensemble.score_batch(test_sessions)
    y_scores      = np.array([r["risk_score"]      for r in scores_list])
    y_true        = np.array([s.label               for s in test_sessions])
    test_classes  = [s.traffic_class.value          for s in test_sessions]

    opt_thr, opt_metrics = find_optimal_threshold(y_true, y_scores, target_fpr=args.target_fpr)
    per_cls = per_class_metrics(test_classes, y_true, y_scores, threshold=opt_thr)
    print_eval_report(opt_metrics, per_cls, opt_thr, label=f"{args.data_source.upper()} Ensemble")

    print("  Top 10 features (Gradient Boosting):")
    for feat, imp in gb_det.top_features(10):
        print(f"    {feat:<42} {imp:.4f}  {'█'*int(imp*200)}")

    results = dict(data_source=args.data_source, n_sessions=len(sessions),
                   train_size=len(train_ds), test_size=len(test_ds),
                   optimal_threshold=opt_thr, metrics=opt_metrics, per_class=per_cls)
    out_path = os.path.join(args.output_dir, "train_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results → {out_path}")
    print(f"  Training complete!")


if __name__ == "__main__":
    main()
