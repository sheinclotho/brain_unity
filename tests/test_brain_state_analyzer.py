"""Unit tests for BrainStateAnalyzer"""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from unity_integration.brain_state_analyzer import BrainStateAnalyzer


class TestComputeDeviationMap(unittest.TestCase):
    def _make_ts(self, T, N, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((T, N)).astype(np.float32)

    def test_output_shape(self):
        ref  = self._make_ts(50, 10)
        test = self._make_ts(50, 10, seed=1)
        overlay, summary = BrainStateAnalyzer.compute_deviation_map(ref, test)
        self.assertEqual(overlay.shape, (10,))
        self.assertEqual(overlay.dtype, np.float32)

    def test_output_range(self):
        """Normalised overlay should lie in [0, 1]."""
        ref  = self._make_ts(40, 20)
        test = self._make_ts(40, 20, seed=2)
        overlay, _ = BrainStateAnalyzer.compute_deviation_map(ref, test)
        self.assertGreaterEqual(float(overlay.min()), 0.0)
        self.assertLessEqual(float(overlay.max()), 1.0 + 1e-6)

    def test_summary_keys(self):
        ref  = self._make_ts(30, 5)
        test = self._make_ts(30, 5, seed=3)
        _, summary = BrainStateAnalyzer.compute_deviation_map(ref, test)
        for key in ("mean_cohens_d", "max_cohens_d", "n_large_effect",
                    "n_activated", "n_suppressed", "interpretation",
                    "effect_size_metric"):
            self.assertIn(key, summary, f"Missing summary key: {key}")

    def test_insufficient_samples_returns_zeros(self):
        """Windows with fewer than 2 frames should return zero overlay."""
        ref  = self._make_ts(1, 10)
        test = self._make_ts(1, 10, seed=4)
        overlay, summary = BrainStateAnalyzer.compute_deviation_map(ref, test)
        np.testing.assert_array_equal(overlay, np.zeros(10, dtype=np.float32))
        self.assertEqual(summary["mean_cohens_d"], 0.0)

    def test_identical_windows_near_zero_effect(self):
        """Same data in both windows → Cohen's d ≈ 0 for all regions."""
        ts = self._make_ts(40, 10)
        overlay, summary = BrainStateAnalyzer.compute_deviation_map(ts, ts)
        self.assertAlmostEqual(summary["mean_cohens_d"], 0.0, places=5)

    def test_effect_size_metric_label(self):
        ref  = self._make_ts(20, 5)
        test = self._make_ts(20, 5, seed=5)
        _, summary = BrainStateAnalyzer.compute_deviation_map(ref, test)
        self.assertIn("Cohen", summary["effect_size_metric"])


class TestComputeGraphMetrics(unittest.TestCase):
    def _make_ec(self, N=20, seed=0):
        rng = np.random.default_rng(seed)
        ec = rng.random((N, N)).astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        return ec / ec.max()  # normalise to [0, 1]

    def test_output_keys(self):
        ec = self._make_ec()
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec)
        for key in ("hub_scores", "out_strength", "in_strength",
                    "local_eff", "global_eff", "density", "top_hubs"):
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_hub_scores_length(self):
        N = 30
        ec = self._make_ec(N)
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec)
        self.assertEqual(len(metrics["hub_scores"]), N)

    def test_density_range(self):
        ec = self._make_ec()
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec)
        self.assertGreaterEqual(metrics["density"], 0.0)
        self.assertLessEqual(metrics["density"], 1.0)

    def test_top_hubs_are_valid_indices(self):
        N = 20
        ec = self._make_ec(N)
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec)
        for idx in metrics["top_hubs"]:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, N)

    def test_zero_ec_returns_zero_density(self):
        ec = np.zeros((10, 10), dtype=np.float32)
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec)
        self.assertEqual(metrics["density"], 0.0)
        self.assertEqual(metrics["global_eff"], 0.0)


class TestCompareECMatrices(unittest.TestCase):
    def _make_ec(self, N=15, seed=0):
        rng = np.random.default_rng(seed)
        ec = rng.random((N, N)).astype(np.float32) * 0.5
        np.fill_diagonal(ec, 0.0)
        return ec

    def test_identical_matrices_high_similarity(self):
        ec = self._make_ec()
        overlay, summary = BrainStateAnalyzer.compare_ec_matrices(ec, ec)
        self.assertAlmostEqual(summary["pearson_r"], 1.0, places=4)

    def test_output_shape(self):
        N = 15
        ec1 = self._make_ec(N, seed=0)
        ec2 = self._make_ec(N, seed=1)
        overlay, summary = BrainStateAnalyzer.compare_ec_matrices(ec1, ec2)
        self.assertEqual(len(overlay), N)

    def test_overlay_range(self):
        ec1 = self._make_ec(seed=0)
        ec2 = self._make_ec(seed=2)
        overlay, _ = BrainStateAnalyzer.compare_ec_matrices(ec1, ec2)
        self.assertGreaterEqual(float(overlay.min()), 0.0)
        self.assertLessEqual(float(overlay.max()), 1.0 + 1e-6)


class TestEcVsDistanceCorrelation(unittest.TestCase):
    def test_runs_without_error(self):
        rng = np.random.default_rng(42)
        ec = rng.random((20, 20)).astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        result = BrainStateAnalyzer.ec_vs_distance_correlation(ec)
        self.assertIn("ec_vs_distance_r", result)
        self.assertIn("interpretation", result)

    def test_uses_cached_positions_consistently(self):
        """Two calls with the same EC should return the same r value."""
        rng = np.random.default_rng(7)
        ec = rng.random((20, 20)).astype(np.float32)
        np.fill_diagonal(ec, 0.0)
        r1 = BrainStateAnalyzer.ec_vs_distance_correlation(ec)["ec_vs_distance_r"]
        r2 = BrainStateAnalyzer.ec_vs_distance_correlation(ec)["ec_vs_distance_r"]
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    unittest.main()
