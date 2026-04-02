"""
tests/test_all.py — Unit tests for the Bot Detection System.

Run:
    python -m pytest tests/ -v
    python tests/test_all.py

Groups:
  A — Config
  B — Traffic simulator
  C — HTTP feature extraction
  D — Behavioural feature extraction
  E — LLM fingerprint signals
  F — Feature pipeline (end-to-end)
  G — Statistical detector
  H — ML detectors (IF + GB)
  I — LLM detector
  J — Ensemble
  K — Mitigation engine
  L — Evaluation metrics
"""

import sys, os, math, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import Config
from data.simulator import TrafficSimulator, TrafficClass, Session, Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    cfg = Config()
    cfg.simulation.n_sessions = 200
    return cfg

def _sim(n=200, seed=42):
    cfg = _cfg()
    cfg.simulation.n_sessions = n
    return TrafficSimulator(cfg.simulation, seed=seed).generate()

def _one(tc: TrafficClass, seed=0):
    cfg = _cfg()
    cfg.simulation.n_sessions = 20
    sim = TrafficSimulator(cfg.simulation, seed=seed)
    for s in sim.generate():
        if s.traffic_class == tc:
            return s
    raise ValueError(f"No session of type {tc}")


# =============================================================================
# A — Config
# =============================================================================
class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = Config()
        self.assertGreater(cfg.statistical.rate_block_rpm, cfg.statistical.rate_warn_rpm)
        self.assertIn("statistical", cfg.ensemble.weights)
        self.assertEqual(len(cfg.mitigation.thresholds), 5)

    def test_mitigation_thresholds_cover_01(self):
        cfg = Config()
        all_ranges = list(cfg.mitigation.thresholds.values())
        self.assertAlmostEqual(all_ranges[0][0], 0.0)
        self.assertAlmostEqual(all_ranges[-1][1], 1.0)

    def test_ensemble_weights_sum_to_1(self):
        cfg = Config()
        s = sum(cfg.ensemble.weights.values())
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_summary_contains_sections(self):
        s = Config().summary()
        for sec in ("STATISTICAL", "ML", "LLM", "ENSEMBLE", "MITIGATION"):
            self.assertIn(sec, s)


# =============================================================================
# B — Traffic simulator
# =============================================================================
class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.sessions = _sim(200)

    def test_correct_number_of_sessions(self):
        self.assertEqual(len(self.sessions), 200)

    def test_all_classes_present(self):
        classes = {s.traffic_class for s in self.sessions}
        for tc in TrafficClass:
            self.assertIn(tc, classes, f"Missing class: {tc}")

    def test_human_high_iat_variance(self):
        humans = [s for s in self.sessions if s.traffic_class == TrafficClass.HUMAN]
        cvs = []
        for s in humans:
            iats = s.inter_arrival_times
            if len(iats) < 3:
                continue
            mean = sum(iats) / len(iats)
            std  = (sum((x-mean)**2 for x in iats) / len(iats)) ** 0.5
            cvs.append(std / max(mean, 1.0))
        mean_cv = sum(cvs) / max(len(cvs), 1)
        self.assertGreater(mean_cv, 0.5, "Humans should have high IAT variance")

    def test_llm_agent_low_iat_variance(self):
        llm = [s for s in self.sessions if s.traffic_class == TrafficClass.LLM_AGENT]
        cvs = []
        for s in llm:
            iats = s.inter_arrival_times
            if len(iats) < 3:
                continue
            mean = sum(iats) / len(iats)
            std  = (sum((x-mean)**2 for x in iats) / len(iats)) ** 0.5
            cvs.append(std / max(mean, 1.0))
        mean_cv = sum(cvs) / max(len(cvs), 1)
        self.assertLess(mean_cv, 0.5, "LLM agents should have low IAT variance")

    def test_labels_correct(self):
        for s in self.sessions:
            if s.traffic_class == TrafficClass.HUMAN:
                self.assertEqual(s.label, 0)
            else:
                self.assertEqual(s.label, 1)

    def test_session_has_requests(self):
        for s in self.sessions:
            self.assertGreater(len(s.requests), 0)

    def test_cred_stuffer_mostly_posts(self):
        stuffers = [s for s in self.sessions if s.traffic_class == TrafficClass.CRED_STUFFER]
        for s in stuffers[:3]:
            posts = sum(1 for r in s.requests if r.method == "POST")
            self.assertGreater(posts / len(s.requests), 0.5)


# =============================================================================
# C — HTTP Features
# =============================================================================
class TestHTTPFeatures(unittest.TestCase):
    def test_feature_keys(self):
        from features.http_features import extract_http_features
        s = _one(TrafficClass.HUMAN)
        f = extract_http_features(s)
        for key in ("iat_cv", "rpm_mean", "error_rate", "url_entropy",
                    "header_completeness", "session_depth"):
            self.assertIn(key, f)

    def test_all_values_finite(self):
        from features.http_features import extract_http_features
        for tc in TrafficClass:
            s = _one(tc)
            f = extract_http_features(s)
            for k, v in f.items():
                self.assertTrue(math.isfinite(v), f"Non-finite {k}={v} for {tc}")

    def test_bot_ua_no_sec_fetch(self):
        from features.http_features import extract_http_features
        s = _one(TrafficClass.SIMPLE_BOT)
        f = extract_http_features(s)
        self.assertEqual(f["has_sec_fetch"], 0.0)

    def test_human_has_full_headers(self):
        from features.http_features import extract_http_features
        s = _one(TrafficClass.HUMAN)
        f = extract_http_features(s)
        self.assertGreater(f["header_completeness"], 0.5)

    def test_rpm_positive(self):
        from features.http_features import extract_http_features
        s = _one(TrafficClass.SIMPLE_BOT)
        f = extract_http_features(s)
        self.assertGreater(f["rpm_mean"], 0)


# =============================================================================
# D — Behavioural Features
# =============================================================================
class TestBehavioralFeatures(unittest.TestCase):
    def test_feature_keys(self):
        from features.behavioral_features import extract_behavioral_features
        s = _one(TrafficClass.HUMAN)
        f = extract_behavioral_features(s)
        for key in ("dwell_cv", "backtrack_rate", "path_entropy",
                    "session_linearity", "api_rate"):
            self.assertIn(key, f)

    def test_all_values_finite(self):
        from features.behavioral_features import extract_behavioral_features
        for tc in TrafficClass:
            s = _one(tc)
            f = extract_behavioral_features(s)
            for k, v in f.items():
                self.assertTrue(math.isfinite(v), f"Non-finite {k} for {tc}")

    def test_llm_high_session_linearity(self):
        from features.behavioral_features import extract_behavioral_features
        s = _one(TrafficClass.LLM_AGENT)
        f = extract_behavioral_features(s)
        self.assertGreater(f["session_linearity"], 0.5)

    def test_human_higher_dwell_cv_than_bot(self):
        from features.behavioral_features import extract_behavioral_features
        h = _one(TrafficClass.HUMAN)
        b = _one(TrafficClass.SIMPLE_BOT)
        fh = extract_behavioral_features(h)
        fb = extract_behavioral_features(b)
        self.assertGreater(fh["dwell_cv"], fb["dwell_cv"])


# =============================================================================
# E — LLM Fingerprints
# =============================================================================
class TestLLMFingerprints(unittest.TestCase):
    def test_timing_regularity_llm_higher_than_human(self):
        from features.llm_fingerprints import timing_regularity_score
        h   = _one(TrafficClass.HUMAN)
        llm = _one(TrafficClass.LLM_AGENT)
        self.assertGreater(
            timing_regularity_score(llm),
            timing_regularity_score(h),
        )

    def test_ua_consistency_detects_bot_ua(self):
        from features.llm_fingerprints import ua_consistency_score
        s = _one(TrafficClass.SIMPLE_BOT)
        score = ua_consistency_score(s)
        self.assertGreater(score, 0.5)

    def test_composite_llm_score_range(self):
        from features.llm_fingerprints import compute_llm_fingerprint
        cfg = _cfg()
        for tc in TrafficClass:
            s   = _one(tc)
            out = compute_llm_fingerprint(s, cfg)
            self.assertIn("llm_score", out)
            self.assertGreaterEqual(out["llm_score"], 0.0)
            self.assertLessEqual(out["llm_score"], 1.0)

    def test_llm_agent_highest_composite_score(self):
        from features.llm_fingerprints import compute_llm_fingerprint
        cfg = _cfg()
        scores = {}
        for tc in TrafficClass:
            s = _one(tc)
            scores[tc] = compute_llm_fingerprint(s, cfg)["llm_score"]
        self.assertGreater(scores[TrafficClass.LLM_AGENT], scores[TrafficClass.HUMAN])


# =============================================================================
# F — Feature Pipeline
# =============================================================================
class TestFeaturePipeline(unittest.TestCase):
    def test_build_dataset_shape(self):
        from features.feature_pipeline import build_dataset
        sessions = _sim(100)
        ds = build_dataset(sessions, _cfg())
        self.assertEqual(ds.X.shape[0], 100)
        self.assertGreater(ds.X.shape[1], 20)

    def test_no_nan_in_features(self):
        from features.feature_pipeline import build_dataset
        sessions = _sim(100)
        ds = build_dataset(sessions, _cfg())
        self.assertFalse(np.isnan(ds.X).any())
        self.assertFalse(np.isinf(ds.X).any())

    def test_labels_binary(self):
        from features.feature_pipeline import build_dataset
        sessions = _sim(100)
        ds = build_dataset(sessions, _cfg())
        self.assertTrue(set(ds.y_binary).issubset({0, 1}))

    def test_train_test_split(self):
        from features.feature_pipeline import build_dataset
        sessions = _sim(200)
        ds = build_dataset(sessions, _cfg())
        train, test = ds.train_test_split(test_fraction=0.2)
        self.assertAlmostEqual(len(test) / len(ds), 0.2, delta=0.05)
        self.assertEqual(len(train) + len(test), len(ds))


# =============================================================================
# G — Statistical Detector
# =============================================================================
class TestStatisticalDetector(unittest.TestCase):
    def setUp(self):
        self.det = StatisticalDetector(_cfg())

    def test_score_range(self):
        from detectors.statistical import StatisticalDetector
        det = StatisticalDetector(_cfg())
        for tc in TrafficClass:
            s = _one(tc)
            score, _ = det.score(s)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_bot_ua_scores_high(self):
        from detectors.statistical import StatisticalDetector
        det = StatisticalDetector(_cfg())
        s   = _one(TrafficClass.SIMPLE_BOT)
        score, reason = det.score(s)
        self.assertGreater(score, 0.5)

    def test_human_scores_low(self):
        from detectors.statistical import StatisticalDetector
        det = StatisticalDetector(_cfg())
        # Average over several human sessions
        scores = [det.score(_one(TrafficClass.HUMAN, seed=i))[0] for i in range(5)]
        mean_score = sum(scores) / len(scores)
        self.assertLess(mean_score, 0.6)

    def test_reason_is_string(self):
        from detectors.statistical import StatisticalDetector
        det = StatisticalDetector(_cfg())
        _, reason = det.score(_one(TrafficClass.HUMAN))
        self.assertIsInstance(reason, str)

    def test_predict_returns_binary(self):
        from detectors.statistical import StatisticalDetector
        det = StatisticalDetector(_cfg())
        for tc in TrafficClass:
            pred = det.predict(_one(tc))
            self.assertIn(pred, (0, 1))


# =============================================================================
# H — ML Detectors
# =============================================================================
class TestMLDetectors(unittest.TestCase):
    def setUp(self):
        from features.feature_pipeline import build_dataset
        sessions  = _sim(300)
        self.ds   = build_dataset(sessions, _cfg())
        self.cfg  = _cfg()

    def test_isolation_forest_fit_score(self):
        from detectors.ml_detector import IsolationForestDetector
        det = IsolationForestDetector(self.cfg)
        det.fit(self.ds)
        scores = det.score(self.ds.X)
        self.assertEqual(scores.shape[0], len(self.ds))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())

    def test_gradient_boosting_fit_predict(self):
        from detectors.ml_detector import GradientBoostingDetector
        det = GradientBoostingDetector(self.cfg)
        det.fit(self.ds)
        preds = det.predict(self.ds.X)
        self.assertEqual(preds.shape[0], len(self.ds))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_gb_accuracy_above_random(self):
        from detectors.ml_detector import GradientBoostingDetector
        det = GradientBoostingDetector(self.cfg)
        det.fit(self.ds)
        preds = det.predict(self.ds.X)
        acc   = (preds == self.ds.y_binary).mean()
        self.assertGreater(acc, 0.6)

    def test_top_features_returns_list(self):
        from detectors.ml_detector import GradientBoostingDetector
        det = GradientBoostingDetector(self.cfg)
        det.fit(self.ds)
        top = det.top_features(5)
        self.assertEqual(len(top), 5)
        for name, imp in top:
            self.assertIsInstance(name, str)
            self.assertGreaterEqual(imp, 0.0)

    def test_save_load(self):
        import tempfile
        from detectors.ml_detector import GradientBoostingDetector
        det = GradientBoostingDetector(self.cfg)
        det.fit(self.ds)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            det.save(path)
            det2 = GradientBoostingDetector.load(path, self.cfg)
            s1 = det.score(self.ds.X[:5])
            s2 = det2.score(self.ds.X[:5])
            np.testing.assert_allclose(s1, s2, rtol=1e-5)
        finally:
            os.unlink(path)


# =============================================================================
# I — LLM Detector
# =============================================================================
class TestLLMDetector(unittest.TestCase):
    def setUp(self):
        from detectors.llm_detector import LLMAgentDetector
        self.det = LLMAgentDetector(_cfg())

    def test_score_returns_tuple(self):
        s = _one(TrafficClass.LLM_AGENT)
        score, signals, conf = self.det.score(s)
        self.assertIsInstance(score, float)
        self.assertIsInstance(signals, dict)
        self.assertIn(conf, ("LOW", "MEDIUM", "HIGH", "INSUFFICIENT_DATA"))

    def test_score_in_range(self):
        for tc in TrafficClass:
            score, _, _ = self.det.score(_one(tc))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_llm_agent_scores_higher_than_human(self):
        llm_score, _, _ = self.det.score(_one(TrafficClass.LLM_AGENT))
        hum_score, _, _ = self.det.score(_one(TrafficClass.HUMAN))
        self.assertGreater(llm_score, hum_score)

    def test_explain_returns_string(self):
        s     = _one(TrafficClass.LLM_AGENT)
        expl  = self.det.explain(s)
        self.assertIsInstance(expl, str)
        self.assertIn("LLM", expl)


# =============================================================================
# J — Ensemble
# =============================================================================
class TestEnsemble(unittest.TestCase):
    def _build(self):
        from features.feature_pipeline import build_dataset
        from detectors.ml_detector import IsolationForestDetector, GradientBoostingDetector
        from detectors.llm_detector import LLMAgentDetector
        from detectors.statistical import StatisticalDetector
        from detectors.ensemble import EnsembleDetector
        cfg      = _cfg()
        sessions = _sim(300)
        ds       = build_dataset(sessions, cfg)
        if_det   = IsolationForestDetector(cfg); if_det.fit(ds)
        gb_det   = GradientBoostingDetector(cfg); gb_det.fit(ds)
        return EnsembleDetector(cfg, if_det, gb_det, LLMAgentDetector(cfg), StatisticalDetector(cfg))

    def test_score_session_keys(self):
        ens = self._build()
        s   = _one(TrafficClass.LLM_AGENT)
        r   = ens.score_session(s)
        for key in ("risk_score","statistical_score","gb_score","llm_score","n_detectors_firing"):
            self.assertIn(key, r)

    def test_risk_score_in_range(self):
        ens = self._build()
        for tc in TrafficClass:
            r = ens.score_session(_one(tc))
            self.assertGreaterEqual(r["risk_score"], 0.0)
            self.assertLessEqual(r["risk_score"], 1.0)

    def test_batch_same_as_individual(self):
        ens      = self._build()
        sessions = [_one(TrafficClass.HUMAN), _one(TrafficClass.LLM_AGENT)]
        batch    = ens.score_batch(sessions)
        for i, s in enumerate(sessions):
            single = ens.score_session(s)
            self.assertAlmostEqual(batch[i]["risk_score"], single["risk_score"], places=5)


# =============================================================================
# K — Mitigation
# =============================================================================
class TestMitigation(unittest.TestCase):
    def setUp(self):
        from mitigation.strategies import MitigationEngine
        self.engine = MitigationEngine(_cfg())

    def _fake_result(self, score):
        return {"risk_score": score, "explanation": "test", "session_id": "abc123"}

    def test_low_score_allows(self):
        from mitigation.strategies import Action
        d = self.engine.decide(self._fake_result(0.1))
        self.assertEqual(d.action, Action.ALLOW)

    def test_high_score_blocks(self):
        from mitigation.strategies import Action
        d = self.engine.decide(self._fake_result(0.95))
        self.assertEqual(d.action, Action.BLOCK)

    def test_medium_score_challenges_or_throttles(self):
        from mitigation.strategies import Action
        d = self.engine.decide(self._fake_result(0.75))
        self.assertIn(d.action, (Action.CHALLENGE, Action.THROTTLE))

    def test_block_has_ttl(self):
        d = self.engine.decide(self._fake_result(0.95))
        self.assertGreater(d.block_ttl_s, 0)

    def test_action_summary(self):
        from mitigation.strategies import Action
        decisions = [
            self.engine.decide(self._fake_result(s))
            for s in [0.1, 0.4, 0.65, 0.78, 0.92]
        ]
        summary = self.engine.action_summary(decisions)
        self.assertIsInstance(summary, dict)
        self.assertEqual(sum(summary.values()), 5)


# =============================================================================
# L — Evaluation Metrics
# =============================================================================
class TestEvaluationMetrics(unittest.TestCase):
    def test_perfect_classifier(self):
        from evaluation.metrics import compute_metrics
        y = np.array([0,0,1,1,1])
        m = compute_metrics(y, None, y_scores=np.array([0.0,0.1,0.9,0.95,0.8]))
        self.assertAlmostEqual(m["precision"], 1.0, places=4)
        self.assertAlmostEqual(m["recall"],    1.0, places=4)
        self.assertAlmostEqual(m["fpr"],       0.0, places=4)

    def test_random_classifier_auroc_near_half(self):
        from evaluation.metrics import compute_metrics
        np.random.seed(0)
        y = np.random.randint(0, 2, 500)
        s = np.random.uniform(0, 1, 500)
        m = compute_metrics(y, None, y_scores=s)
        self.assertAlmostEqual(m["auroc"], 0.5, delta=0.08)

    def test_find_optimal_threshold(self):
        from evaluation.metrics import find_optimal_threshold
        np.random.seed(0)
        y = np.array([0]*700 + [1]*300)
        s = np.concatenate([np.random.uniform(0, 0.4, 700),
                            np.random.uniform(0.6, 1.0, 300)])
        thr, m = find_optimal_threshold(y, s, target_fpr=0.01)
        self.assertLessEqual(m["fpr"], 0.05)   # allow small slack
        self.assertGreater(m["recall"], 0.5)

    def test_per_class_metrics_keys(self):
        from evaluation.metrics import per_class_metrics
        classes = ["human"]*5 + ["llm_agent"]*5
        y = np.array([0]*5 + [1]*5)
        s = np.array([0.1]*5 + [0.9]*5)
        result = per_class_metrics(classes, y, s)
        self.assertIn("human",     result)
        self.assertIn("llm_agent", result)
        self.assertIn("fpr",       result["human"])
        self.assertIn("recall",    result["llm_agent"])


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
