"""Unit tests for PerturbationAnalyzer and _build_lag_dataset"""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from unity_integration.perturbation_analyzer import _build_lag_dataset, PerturbationAnalyzer


class TestBuildLagDataset(unittest.TestCase):
    def test_output_shape(self):
        T, N, n_lags = 20, 5, 3
        ts = np.random.rand(T, N).astype(np.float32)
        X, Y = _build_lag_dataset(ts, n_lags)
        self.assertEqual(X.shape, (T - n_lags, N * n_lags))
        self.assertEqual(Y.shape, (T - n_lags, N))

    def test_output_dtype(self):
        ts = np.random.rand(10, 4).astype(np.float64)
        X, Y = _build_lag_dataset(ts, 2)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(Y.dtype, np.float32)

    def test_values_correct(self):
        """Vectorised output must match the reference element-by-element."""
        T, N, n_lags = 15, 3, 4
        rng = np.random.default_rng(0)
        ts = rng.random((T, N)).astype(np.float32)

        # Reference implementation (original loop-based approach)
        M = T - n_lags
        X_ref = np.zeros((M, N * n_lags), dtype=np.float32)
        Y_ref = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            X_ref[i] = ts[i: i + n_lags].flatten()
            Y_ref[i] = ts[i + n_lags]

        X, Y = _build_lag_dataset(ts, n_lags)
        np.testing.assert_allclose(X, X_ref, atol=1e-6,
                                   err_msg="X values mismatch vs reference loop")
        np.testing.assert_allclose(Y, Y_ref, atol=1e-6,
                                   err_msg="Y values mismatch vs reference loop")

    def test_n_lags_one(self):
        T, N = 10, 6
        ts = np.arange(T * N, dtype=np.float32).reshape(T, N)
        X, Y = _build_lag_dataset(ts, 1)
        self.assertEqual(X.shape, (T - 1, N))
        np.testing.assert_array_equal(X, ts[:-1])
        np.testing.assert_array_equal(Y, ts[1:])

    def test_handles_large_dataset(self):
        """Vectorised version should handle 2000×200 without error."""
        ts = np.random.rand(2000, 200).astype(np.float32)
        X, Y = _build_lag_dataset(ts, 5)
        self.assertEqual(X.shape, (1995, 200 * 5))
        self.assertEqual(Y.shape, (1995, 200))


class TestPerturbationAnalyzerInit(unittest.TestCase):
    def test_default_init(self):
        analyzer = PerturbationAnalyzer(n_regions=200, n_lags=5)
        self.assertEqual(analyzer.n_regions, 200)
        self.assertEqual(analyzer.n_lags, 5)
        self.assertIsNone(analyzer._surrogate)
        self.assertIsNone(analyzer._input_X)
        self.assertIsNone(analyzer._last_ec)

    def test_custom_init(self):
        analyzer = PerturbationAnalyzer(n_regions=50, n_lags=3)
        self.assertEqual(analyzer.n_regions, 50)
        self.assertEqual(analyzer.n_lags, 3)

    def test_fit_surrogate_smoke(self):
        """fit_surrogate on a tiny dataset should complete without crashing."""
        rng = np.random.default_rng(1)
        ts = rng.random((40, 10)).astype(np.float32)
        analyzer = PerturbationAnalyzer(n_regions=10, n_lags=2)
        train_losses, val_losses = analyzer.fit_surrogate(
            ts, n_lags=2, num_epochs=3, batch_size=8
        )
        self.assertIsNotNone(analyzer._surrogate)
        self.assertIsNotNone(analyzer._input_X)
        self.assertIsInstance(train_losses, list)
        self.assertIsInstance(val_losses, list)

    def test_fit_quality_populated(self):
        rng = np.random.default_rng(2)
        ts = rng.random((30, 8)).astype(np.float32)
        analyzer = PerturbationAnalyzer(n_regions=8, n_lags=2)
        analyzer.fit_surrogate(ts, n_lags=2, num_epochs=2, batch_size=8)
        fq = analyzer._fit_quality
        self.assertIn("train_mse", fq)
        self.assertIn("val_mse", fq)
        self.assertIn("reliable", fq)

    def test_infer_ec_demo(self):
        analyzer = PerturbationAnalyzer(n_regions=200, n_lags=2)
        ec = analyzer.infer_ec_demo()
        self.assertEqual(ec.shape, (200, 200))

    def test_ec_to_dict(self):
        analyzer = PerturbationAnalyzer(n_regions=200, n_lags=2)
        ec = analyzer.infer_ec_demo()
        result = analyzer.ec_to_dict(ec)
        self.assertIn("ec_flat", result)
        self.assertIn("top_sources", result)
        self.assertIn("top_targets", result)
        self.assertEqual(len(result["ec_flat"]), 200 * 200)


class TestComputeResponseMatrix(unittest.TestCase):
    """Tests for compute_response_matrix, analyze_response_matrix,
    and validate_response_matrix."""

    @classmethod
    def setUpClass(cls):
        """Train a small surrogate once for all tests in this class."""
        rng = np.random.default_rng(42)
        # 60 time steps, 10 regions — fast enough for unit tests
        cls.N = 10
        cls.ts = rng.random((60, cls.N)).astype(np.float32)
        cls.analyzer = PerturbationAnalyzer(n_regions=cls.N, n_lags=2)
        cls.analyzer.fit_surrogate(cls.ts, n_lags=2, num_epochs=5, batch_size=16)
        cls.init_state = cls.ts[0].copy()

    # ── compute_response_matrix ─────────────────────────────────────────

    def test_output_shape_sustained(self):
        """R should be (n_stim, N, T)."""
        stim_regions = [0, 1, 2]
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=stim_regions,
            rollout_steps=8,
            sustained_steps=3,
            alpha=0.3,
            mode="sustained",
        )
        R = result["R"]
        self.assertEqual(R.shape, (len(stim_regions), self.N, 8))
        self.assertIsInstance(R, np.ndarray)
        self.assertEqual(R.dtype, np.float32)

    def test_output_shape_impulse(self):
        """Impulse mode should also return correct shape."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=[0],
            rollout_steps=5,
            mode="impulse",
        )
        self.assertEqual(result["R"].shape, (1, self.N, 5))
        self.assertEqual(result["mode"], "impulse")
        self.assertEqual(result["sustained_steps"], 1)

    def test_baseline_included(self):
        """Result dict must include baseline trajectory (N, T)."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=[0],
            rollout_steps=6,
        )
        bl = result["baseline"]
        self.assertEqual(bl.shape, (self.N, 6))

    def test_zero_alpha_gives_near_zero_R(self):
        """With alpha=0 there is no perturbation so ΔX ≈ 0."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=[0],
            rollout_steps=5,
            alpha=0.0,
        )
        np.testing.assert_allclose(
            result["R"], 0.0, atol=1e-6,
            err_msg="alpha=0 should give zero response"
        )

    def test_nonzero_alpha_gives_nonzero_R(self):
        """With nonzero alpha, at least the stimulated region should respond."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=[0],
            rollout_steps=5,
            alpha=0.3,
        )
        # Mean absolute response should be positive
        self.assertGreater(float(np.abs(result["R"]).mean()), 0.0)

    def test_sustained_vs_impulse_differ(self):
        """Sustained and impulse modes should yield different R matrices."""
        common = dict(
            initial_state=self.init_state, stim_regions=[0],
            rollout_steps=8, sustained_steps=5, alpha=0.3,
        )
        R_sus = self.analyzer.compute_response_matrix(
            **common, mode="sustained"
        )["R"]
        R_imp = self.analyzer.compute_response_matrix(
            **common, mode="impulse"
        )["R"]
        # They should differ beyond numerical noise
        self.assertFalse(np.allclose(R_sus, R_imp, atol=1e-5))

    def test_default_all_regions(self):
        """When stim_regions=None all N regions are stimulated."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state,
            stim_regions=None,
            rollout_steps=4,
        )
        self.assertEqual(result["R"].shape[0], self.N)
        self.assertEqual(len(result["stim_regions"]), self.N)

    # ── analyze_response_matrix ─────────────────────────────────────────

    def test_analyze_output_keys(self):
        """analyze_response_matrix must return all required keys."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state, stim_regions=[0, 1],
            rollout_steps=8,
        )
        analysis = PerturbationAnalyzer.analyze_response_matrix(
            result["R"], result["stim_regions"], N=self.N
        )
        for key in (
            "spatial_spread", "temporal_decay", "delay_peak_steps",
            "off_diagonal_ratio", "mean_response_map",
            "has_spatial_spread", "has_decay", "has_delay",
            "plausibility_summary",
        ):
            self.assertIn(key, analysis, f"Missing key: {key}")

    def test_analyze_ranges(self):
        """Scalar metrics should lie in expected ranges."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state, stim_regions=[0, 1],
            rollout_steps=10, alpha=0.3,
        )
        analysis = PerturbationAnalyzer.analyze_response_matrix(
            result["R"], result["stim_regions"], N=self.N
        )
        self.assertGreaterEqual(analysis["spatial_spread"], 0.0)
        self.assertLessEqual(analysis["spatial_spread"], 1.0)
        self.assertIsInstance(analysis["temporal_decay"], float)
        self.assertEqual(
            len(analysis["delay_peak_steps"]), len(result["stim_regions"])
        )
        self.assertGreaterEqual(analysis["off_diagonal_ratio"], 0.0)
        self.assertEqual(len(analysis["mean_response_map"]), self.N)

    def test_analyze_plausibility_str(self):
        """plausibility_summary must be a non-empty string."""
        result = self.analyzer.compute_response_matrix(
            initial_state=self.init_state, stim_regions=[0],
            rollout_steps=5,
        )
        analysis = PerturbationAnalyzer.analyze_response_matrix(
            result["R"], result["stim_regions"], N=self.N
        )
        self.assertIsInstance(analysis["plausibility_summary"], str)
        self.assertGreater(len(analysis["plausibility_summary"]), 0)

    # ── validate_response_matrix ────────────────────────────────────────

    def test_validate_requires_two_states(self):
        """Single state should return nan and reliable=False."""
        result = self.analyzer.validate_response_matrix(
            initial_states=[self.init_state],
            stim_regions=[0],
        )
        self.assertTrue(np.isnan(result["consistency_r"]))
        self.assertFalse(result["reliable"])

    def test_validate_output_keys(self):
        """validate_response_matrix must return all required keys."""
        rng = np.random.default_rng(7)
        states = [
            rng.random(self.N).astype(np.float32),
            rng.random(self.N).astype(np.float32),
        ]
        result = self.analyzer.validate_response_matrix(
            initial_states=states, stim_regions=[0], rollout_steps=5,
        )
        for key in ("consistency_r", "n_states", "reliable", "interpretation"):
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertEqual(result["n_states"], 2)

    def test_validate_consistency_r_range(self):
        """consistency_r must lie in [-1, 1]."""
        rng = np.random.default_rng(8)
        states = [rng.random(self.N).astype(np.float32) for _ in range(3)]
        result = self.analyzer.validate_response_matrix(
            initial_states=states, stim_regions=[0, 1], rollout_steps=5,
        )
        self.assertGreaterEqual(result["consistency_r"], -1.0 - 1e-6)
        self.assertLessEqual(result["consistency_r"],  1.0 + 1e-6)


if __name__ == '__main__':
    unittest.main()
