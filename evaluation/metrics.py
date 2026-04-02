"""
evaluation/metrics.py — Detection performance metrics.

Metrics for bot detection systems differ from generic classifiers:

  Precision   : Of all sessions we flagged as bots, what fraction was correct?
                (False positives hurt legitimate users → high precision required)

  Recall      : Of all actual bots, what fraction did we catch?
                (False negatives let bots through → high recall required)

  FPR         : Of all legitimate users, what fraction did we wrongly block?
                (The most critical metric for e-commerce; FPR < 1% is the goal)

  FNR         : Of all bots, what fraction did we miss?

  AUPRC       : Area under the Precision-Recall curve.
                More informative than AUROC when classes are imbalanced.

  Per-class   : Precision/recall broken down by bot type (simple, scraper,
                cred_stuffer, llm_agent) — reveals which threats are hardest.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of detection metrics.

    Args:
        y_true    : binary ground truth (0=human, 1=bot)
        y_pred    : binary predictions
        y_scores  : continuous risk scores (for AUROC/AUPRC)
        threshold : decision threshold used

    Returns:
        Dict of metric_name → value
    """
    y_pred_t = (y_scores >= threshold).astype(int) if y_scores is not None else y_pred

    TP = int(((y_true == 1) & (y_pred_t == 1)).sum())
    FP = int(((y_true == 0) & (y_pred_t == 1)).sum())
    TN = int(((y_true == 0) & (y_pred_t == 0)).sum())
    FN = int(((y_true == 1) & (y_pred_t == 0)).sum())

    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)   # = True Positive Rate
    fpr       = FP / max(FP + TN, 1)   # False Positive Rate (most critical!)
    fnr       = FN / max(FN + TP, 1)
    accuracy  = (TP + TN) / max(TP + FP + TN + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "precision":    round(precision, 4),
        "recall":       round(recall,    4),
        "fpr":          round(fpr,       4),
        "fnr":          round(fnr,       4),
        "accuracy":     round(accuracy,  4),
        "f1":           round(f1,        4),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "threshold":    threshold,
    }

    # AUROC and AUPRC if scores available
    if y_scores is not None:
        metrics["auroc"] = round(_auroc(y_true, y_scores), 4)
        metrics["auprc"] = round(_auprc(y_true, y_scores), 4)

    return metrics


def _auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUROC via trapezoidal rule."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tprs = [0.0]; fprs = [0.0]
    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        tp = int(((y_true==1)&(pred==1)).sum())
        fp = int(((y_true==0)&(pred==1)).sum())
        fn = int(((y_true==1)&(pred==0)).sum())
        tn = int(((y_true==0)&(pred==0)).sum())
        tprs.append(tp / max(tp+fn, 1))
        fprs.append(fp / max(fp+tn, 1))
    tprs.append(1.0); fprs.append(1.0)
    return float(np.trapz(tprs, fprs))


def _auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUPRC via trapezoidal rule."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precisions = [1.0]; recalls = [0.0]
    for thr in thresholds:
        pred = (y_scores >= thr).astype(int)
        tp = int(((y_true==1)&(pred==1)).sum())
        fp = int(((y_true==0)&(pred==1)).sum())
        fn = int(((y_true==1)&(pred==0)).sum())
        precisions.append(tp / max(tp+fp, 1))
        recalls.append(tp / max(tp+fn, 1))
    return float(np.trapz(precisions, recalls))


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------

def per_class_metrics(
    traffic_classes: List[str],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute detection performance broken down by bot type.

    Reveals which threats are hardest to detect:
      - simple_bot: usually easy (distinctive headers, rate)
      - llm_agent:  usually hardest (sophisticated, human-like behaviour)
    """
    classes = sorted(set(traffic_classes))
    traffic_classes = np.array(traffic_classes)
    results = {}

    for cls in classes:
        mask = traffic_classes == cls
        if mask.sum() == 0:
            continue
        true_label = 0 if cls == "human" else 1
        y_t = (y_true[mask] == true_label).astype(int)
        # For non-human classes: how many did we catch?
        if cls != "human":
            y_s = y_scores[mask]
            detected = (y_s >= threshold).sum()
            recall   = detected / max(len(y_s), 1)
            results[cls] = {
                "n":       int(mask.sum()),
                "recall":  round(recall, 4),
                "detected": int(detected),
                "missed":  int(len(y_s) - detected),
            }
        else:
            y_s = y_scores[mask]
            false_positives = (y_s >= threshold).sum()
            fpr = false_positives / max(len(y_s), 1)
            results[cls] = {
                "n":             int(mask.sum()),
                "fpr":           round(fpr, 4),
                "false_positives": int(false_positives),
                "correctly_allowed": int(len(y_s) - false_positives),
            }

    return results


# ---------------------------------------------------------------------------
# Optimal threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> Tuple[float, Dict]:
    """
    Find the threshold that maximises recall at a target FPR ≤ target_fpr.

    In production, the goal is:
      "Catch as many bots as possible while blocking < 1% of legitimate users."

    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    best_threshold = 0.5
    best_recall    = 0.0

    for thr in np.linspace(0.1, 0.99, 90):
        m = compute_metrics(y_true, None, y_scores, threshold=float(thr))
        if m["fpr"] <= target_fpr and m["recall"] > best_recall:
            best_recall    = m["recall"]
            best_threshold = float(thr)

    final_metrics = compute_metrics(y_true, None, y_scores, threshold=best_threshold)
    return best_threshold, final_metrics


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_eval_report(
    metrics: Dict,
    per_class: Dict,
    optimal_thr: float,
    label: str = "Ensemble",
):
    """Print a formatted evaluation report to stdout."""
    print(f"\n{'═'*62}")
    print(f"  Evaluation Report — {label}")
    print(f"{'═'*62}")
    print(f"  Threshold (optimal @ FPR≤1%): {optimal_thr:.3f}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'─'*32}")
    for k in ("precision", "recall", "fpr", "fnr", "accuracy", "f1", "auroc", "auprc"):
        if k in metrics:
            v = metrics[k]
            flag = ""
            if k == "fpr" and v > 0.02:
                flag = "  ⚠ HIGH"
            elif k == "recall" and v < 0.90:
                flag = "  ⚠ LOW"
            print(f"  {k:<20} {v:>10.4f}{flag}")
    print(f"  {'─'*32}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")

    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<18} {'N':>6} {'Recall/FPR':>12} {'Detected/FP':>14}")
    print(f"  {'─'*54}")
    for cls, m in per_class.items():
        n = m["n"]
        if cls == "human":
            rate = f"FPR={m['fpr']:.4f}"
            det  = f"FP={m['false_positives']}"
        else:
            rate = f"Recall={m['recall']:.4f}"
            det  = f"{m['detected']}/{n}"
        print(f"  {cls:<18} {n:>6} {rate:>12} {det:>14}")
    print(f"{'═'*62}\n")
