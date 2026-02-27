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


if __name__ == '__main__':
    unittest.main()
