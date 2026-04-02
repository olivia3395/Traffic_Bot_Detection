"""
features/feature_pipeline.py — Feature pipeline: session → feature vector.

Combines HTTP features, behavioural features, and LLM fingerprints
into a single unified feature vector for ML models.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from data.simulator import Session
from data.dataset import BotDetectionDataset
from features.http_features import extract_http_features
from features.behavioral_features import extract_behavioral_features
from features.llm_fingerprints import compute_llm_fingerprint


def extract_all_features(session: Session, cfg) -> Dict[str, float]:
    """Extract all feature groups for one session."""
    http_feats = extract_http_features(session)
    beh_feats  = extract_behavioral_features(session)
    llm_feats  = compute_llm_fingerprint(session, cfg.llm)

    # Merge all feature dicts (prefix to avoid name collisions)
    combined = {}
    for k, v in http_feats.items():
        combined[f"http_{k}"] = float(v)
    for k, v in beh_feats.items():
        combined[f"beh_{k}"] = float(v)
    for k, v in llm_feats.items():
        combined[f"llm_{k}"] = float(v)

    return combined


def build_dataset(sessions: List[Session], cfg) -> BotDetectionDataset:
    """
    Run the feature pipeline over all sessions and build a BotDetectionDataset.

    Args:
        sessions : list of Session objects
        cfg      : Config

    Returns:
        BotDetectionDataset ready for training / evaluation
    """
    from tqdm import tqdm
    from data.simulator import TrafficClass

    multi_label_map = {
        TrafficClass.HUMAN:        0,
        TrafficClass.SIMPLE_BOT:   1,
        TrafficClass.SCRAPER:      2,
        TrafficClass.CRED_STUFFER: 3,
        TrafficClass.LLM_AGENT:    4,
    }

    rows: List[Dict] = []
    y_binary = []
    y_multi  = []
    session_ids    = []
    traffic_classes = []

    print(f"\n[Feature Extraction]  {len(sessions)} sessions")
    for session in tqdm(sessions, desc="  Extracting features"):
        feats = extract_all_features(session, cfg)
        rows.append(feats)
        y_binary.append(session.label)
        y_multi.append(multi_label_map[session.traffic_class])
        session_ids.append(session.session_id)
        traffic_classes.append(session.traffic_class.value)

    # Build numpy arrays
    feature_names = list(rows[0].keys())
    X = np.array([[row[k] for k in feature_names] for row in rows], dtype=np.float32)

    # Replace NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    y_binary = np.array(y_binary, dtype=np.int32)
    y_multi  = np.array(y_multi,  dtype=np.int32)

    print(f"  Feature matrix: {X.shape[0]} sessions × {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(traffic_classes, return_counts=True)))}")

    return BotDetectionDataset(X, y_binary, y_multi, feature_names, session_ids, traffic_classes)
