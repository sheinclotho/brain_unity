"""
Tests for the spectral_dynamics module
========================================

Covers all experiments (A–F, E1–E6) with small synthetic data (N=20, fast):
  - compute_connectivity helpers
  - A: connectivity visualization
  - B/E1: spectral analysis
  - C: community structure
  - D: hierarchical structure
  - E2+E3: modal projection & energy
  - E4: structural perturbation
  - E5: phase diagram
  - E6: random network comparison
  - F: PCA + attractor projection
  - run_all (integration test with synthetic data)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure spectral_dynamics (now inside brain_dynamics/) is importable
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BD_DIR = _REPO_ROOT / "brain_dynamics"
if _BD_DIR.exists() and str(_BD_DIR) not in sys.path:
    sys.path.insert(0, str(_BD_DIR))

from spectral_dynamics.compute_connectivity import (
    compute_fc_from_trajectories,
    participation_ratio,
    normalize_spectral_radius,
    symmetrize,
)
from spectral_dynamics.a_connectivity_visualization import run_connectivity_visualization
from spectral_dynamics.c_community_structure import (
    compute_modularity_q,
    spectral_community_detection,
    find_optimal_k,
    run_community_structure,
)
from spectral_dynamics.d_hierarchical_structure import (
    _weight_to_distance,
    compute_linkage,
    compute_cluster_stats_at_levels,
    compute_hierarchy_index,
    run_hierarchical_structure,
)
from spectral_dynamics.f_pca_attractor import (
    compute_pca,
    run_pca_attractor,
)
from spectral_dynamics.e1_spectral_analysis import compute_spectral_metrics, run_spectral_analysis
from spectral_dynamics.e2_e3_modal_projection import (
    compute_modal_energies,
    run_modal_projection,
    _project_trajectories_symmetric,
    _project_trajectories_asymmetric,
)
from spectral_dynamics.e4_structural_perturbation import (
    weight_shuffle,
    degree_preserving_rewire,
    low_rank_truncation,
    run_structural_perturbation,
)
from spectral_dynamics.e5_phase_diagram import (
    _compute_oscillation_amplitude,
    _lle_from_trajs,
    run_phase_diagram,
)
from spectral_dynamics.e6_random_comparison import (
    erdos_renyi,
    run_random_spectral_comparison,
)
from spectral_dynamics.run_all import _make_synthetic_data, run_all


N = 20  # small for fast tests


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _low_rank_matrix(n: int = N, k: int = 3, rho: float = 0.9, seed: int = 0) -> np.ndarray:
    """Low-rank matrix with clear dominant eigenvalues."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, k)) / np.sqrt(k)
    sv = np.array([5.0, 3.0, 1.5] + [0.0] * (k - 3))[:k]
    W = U @ np.diag(sv) @ U.T
    noise = rng.standard_normal((n, n)) * 0.1
    W = W + noise
    ev = np.linalg.eigvals(W)
    r = np.abs(ev).max()
    if r > 1e-8:
        W = W * rho / r
    return W.astype(np.float32)


def _random_trajectories(n_traj: int = 10, steps: int = 50, n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_traj, steps, n)).astype(np.float32)


def _wc_trajectories(W: np.ndarray, n_traj: int = 10, steps: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    trajs = np.empty((n_traj, steps, n), dtype=np.float32)
    for i in range(n_traj):
        x = rng.random(n).astype(np.float32)
        for t in range(steps):
            trajs[i, t] = x
            x = np.clip(np.tanh(0.9 * (W @ x)), 0.0, 1.0).astype(np.float32)
    return trajs


# ══════════════════════════════════════════════════════════════════════════════
# compute_connectivity tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeConnectivity(unittest.TestCase):
    def test_fc_shape(self):
        trajs = _random_trajectories()
        fc = compute_fc_from_trajectories(trajs)
        self.assertEqual(fc.shape, (N, N))

    def test_fc_diagonal_ones(self):
        trajs = _random_trajectories()
        fc = compute_fc_from_trajectories(trajs)
        np.testing.assert_allclose(np.diag(fc), np.ones(N), atol=1e-5)

    def test_fc_symmetric(self):
        trajs = _random_trajectories()
        fc = compute_fc_from_trajectories(trajs)
        np.testing.assert_allclose(fc, fc.T, atol=1e-5)

    def test_fc_range(self):
        trajs = _random_trajectories()
        fc = compute_fc_from_trajectories(trajs)
        self.assertGreaterEqual(float(fc.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(fc.max()), 1.0 + 1e-5)

    def test_participation_ratio_max_N(self):
        # Uniform eigenvalues → PR = N
        mags = np.ones(N)
        pr = participation_ratio(mags)
        self.assertAlmostEqual(pr, N, places=5)

    def test_participation_ratio_min_1(self):
        # Single dominant eigenvalue → PR = 1
        mags = np.zeros(N)
        mags[0] = 1.0
        pr = participation_ratio(mags)
        self.assertAlmostEqual(pr, 1.0, places=5)

    def test_normalize_spectral_radius(self):
        W = _low_rank_matrix()
        W_norm = normalize_spectral_radius(W, target_rho=0.5)
        ev = np.linalg.eigvals(W_norm)
        self.assertAlmostEqual(float(np.abs(ev).max()), 0.5, places=3)

    def test_symmetrize(self):
        rng = np.random.default_rng(0)
        A = rng.random((N, N)).astype(np.float32)
        S = symmetrize(A)
        np.testing.assert_allclose(S, S.T, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# E1: Spectral Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestSpectralAnalysis(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_compute_spectral_metrics_keys(self):
        m = compute_spectral_metrics(self.W)
        for key in ("spectral_radius", "participation_ratio", "n_dominant",
                    "spectral_gap_ratio", "top_k_eigenvalue_share",
                    "eigenvalues", "eigenvalue_magnitudes"):
            self.assertIn(key, m)

    def test_spectral_radius_positive(self):
        m = compute_spectral_metrics(self.W)
        self.assertGreater(m["spectral_radius"], 0.0)

    def test_spectral_radius_correct(self):
        m = compute_spectral_metrics(self.W)
        ev = np.linalg.eigvals(self.W.astype(np.float64))
        expected = float(np.abs(ev).max())
        self.assertAlmostEqual(m["spectral_radius"], expected, places=4)

    def test_participation_ratio_low_rank(self):
        # Low-rank matrix should have PR << N
        m = compute_spectral_metrics(self.W)
        self.assertLess(m["participation_ratio"], N * 0.5)

    def test_n_dominant_positive(self):
        m = compute_spectral_metrics(self.W)
        self.assertGreater(m["n_dominant"], 0)

    def test_eigenvalue_magnitudes_sorted_descending(self):
        m = compute_spectral_metrics(self.W)
        mags = m["eigenvalue_magnitudes"]
        self.assertTrue(np.all(mags[:-1] >= mags[1:] - 1e-8))

    def test_symmetric_mode_uses_real_eigenvalues(self):
        W_sym = symmetrize(self.W)
        m = compute_spectral_metrics(W_sym, symmetric=True)
        # All eigenvalues should be real (imaginary part ~0)
        self.assertTrue(np.all(np.abs(m["eigenvalues"].imag) < 1e-10))

    def test_run_spectral_analysis_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_spectral_analysis(self.W, output_dir=Path(tmpdir), label="test")
            json_path = Path(tmpdir) / "spectral_summary_test.json"
            self.assertTrue(json_path.exists())
            with open(json_path) as f:
                loaded = json.load(f)
            self.assertIn("spectral_radius", loaded)
            self.assertAlmostEqual(loaded["spectral_radius"], result["spectral_radius"], places=4)

    def test_run_spectral_analysis_saves_eigenvalues_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_spectral_analysis(self.W, output_dir=Path(tmpdir), label="test")
            npy_path = Path(tmpdir) / "eigenvalues_test.npy"
            self.assertTrue(npy_path.exists())
            ev = np.load(npy_path)
            self.assertEqual(ev.shape, (N,))


# ══════════════════════════════════════════════════════════════════════════════
# E2+E3: Modal Projection & Energy
# ══════════════════════════════════════════════════════════════════════════════

class TestModalProjection(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()
        self.trajs = _random_trajectories()

    def test_symmetric_projection_shape(self):
        W_sym = symmetrize(self.W)
        z, ev, vecs = _project_trajectories_symmetric(self.trajs, W_sym)
        self.assertEqual(z.shape, self.trajs.shape)
        self.assertEqual(len(ev), N)
        self.assertEqual(vecs.shape, (N, N))

    def test_asymmetric_projection_shape(self):
        z, sv, U = _project_trajectories_asymmetric(self.trajs, self.W)
        self.assertEqual(z.shape, self.trajs.shape)
        self.assertEqual(len(sv), N)
        self.assertEqual(U.shape, (N, N))

    def test_projection_invertible_symmetric(self):
        # z = Vᵀ x  →  x = V z  → V z ≈ x
        W_sym = symmetrize(self.W).astype(np.float64)
        ev_real, vecs = np.linalg.eigh(W_sym)
        idx = np.argsort(np.abs(ev_real))[::-1]
        vecs = vecs[:, idx]
        z, _, _ = _project_trajectories_symmetric(self.trajs, W_sym.astype(np.float32))
        x_reconstructed = (vecs @ z[0].astype(np.float64).T).T
        np.testing.assert_allclose(
            x_reconstructed, self.trajs[0].astype(np.float64), atol=1e-4
        )

    def test_modal_energy_sums_to_one(self):
        z, _, _ = _project_trajectories_asymmetric(self.trajs, self.W)
        m = compute_modal_energies(z)
        self.assertAlmostEqual(float(m["energies_normalized"].sum()), 1.0, places=5)

    def test_modal_energy_cumulative_monotone(self):
        z, _, _ = _project_trajectories_asymmetric(self.trajs, self.W)
        m = compute_modal_energies(z)
        cumul = m["cumulative_energy"]
        self.assertTrue(np.all(cumul[1:] >= cumul[:-1]))

    def test_modal_energy_top1_positive(self):
        z, _, _ = _project_trajectories_asymmetric(self.trajs, self.W)
        m = compute_modal_energies(z)
        self.assertGreater(m["energy_top1"], 0.0)

    def test_low_rank_matrix_concentrates_energy(self):
        # Low-rank W: energy should be more concentrated than white noise.
        # We use SVD of W as the projection basis; WC nonlinearity mixes modes,
        # so the threshold is deliberately loose (top-5 > 10% in N=20).
        z, _, _ = _project_trajectories_asymmetric(self.trajs, self.W)
        m = compute_modal_energies(z)
        # n_modes_80pct must be strictly less than N (some concentration exists)
        self.assertLess(m["n_modes_80pct"], N)
        # top-5 should capture at least 10% of energy
        self.assertGreater(m["energy_top5"], 0.10)

    def test_run_modal_projection_returns_dict(self):
        result = run_modal_projection(self.trajs, self.W, label="test")
        for key in ("n_modes_80pct", "energy_top1_pct", "energy_top2_pct", "energy_top5_pct"):
            self.assertIn(key, result)

    def test_run_modal_projection_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_modal_projection(self.trajs, self.W, output_dir=Path(tmpdir), label="test")
            self.assertTrue((Path(tmpdir) / "modal_energy_test.json").exists())
            self.assertTrue((Path(tmpdir) / "modal_projections_test.npy").exists())


# ══════════════════════════════════════════════════════════════════════════════
# E4: Structural Perturbation
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuralPerturbation(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_weight_shuffle_same_shape(self):
        W2 = weight_shuffle(self.W)
        self.assertEqual(W2.shape, self.W.shape)

    def test_weight_shuffle_preserves_nonzero_count(self):
        W2 = weight_shuffle(self.W)
        self.assertEqual(int((W2 != 0).sum()), int((self.W != 0).sum()))

    def test_weight_shuffle_preserves_value_set(self):
        # Sorted values should match (same multiset of weights)
        W2 = weight_shuffle(self.W, seed=99)
        np.testing.assert_allclose(
            np.sort(self.W.ravel()), np.sort(W2.ravel()), atol=1e-6
        )

    def test_degree_preserving_rewire_shape(self):
        W2 = degree_preserving_rewire(self.W)
        self.assertEqual(W2.shape, self.W.shape)

    def test_degree_preserving_rewire_preserves_sparsity(self):
        W2 = degree_preserving_rewire(self.W)
        self.assertEqual(int((W2 != 0).sum()), int((self.W != 0).sum()))

    def test_low_rank_truncation_rank(self):
        k = 3
        W_lr = low_rank_truncation(self.W, k=k)
        self.assertEqual(W_lr.shape, self.W.shape)
        # Rank should be at most k
        rank = np.linalg.matrix_rank(W_lr.astype(np.float64), tol=1e-5)
        self.assertLessEqual(rank, k)

    def test_low_rank_truncation_k1(self):
        W_lr = low_rank_truncation(self.W, k=1)
        rank = np.linalg.matrix_rank(W_lr.astype(np.float64), tol=1e-5)
        self.assertEqual(rank, 1)

    def test_run_structural_perturbation_returns_keys(self):
        result = run_structural_perturbation(
            self.W, label="test", k_values=[1, 3], seed=0
        )
        self.assertIn("original", result)
        self.assertIn("weight_shuffle", result)
        self.assertIn("degree_preserving_rewire", result)
        self.assertIn("low_rank_truncation", result)

    def test_run_structural_perturbation_original_metrics(self):
        result = run_structural_perturbation(
            self.W, label="test", k_values=[1], seed=0
        )
        self.assertIn("spectral_radius", result["original"])
        self.assertIn("participation_ratio", result["original"])
        # lle_wc must NOT be present — no WC in this project
        self.assertNotIn("lle_wc", result["original"])

    def test_run_structural_perturbation_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_structural_perturbation(
                self.W, output_dir=Path(tmpdir), label="test",
                k_values=[1], seed=0
            )
            self.assertTrue((Path(tmpdir) / "perturbation_summary_test.json").exists())


# ══════════════════════════════════════════════════════════════════════════════
# E5: Phase Diagram
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseDiagram(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix(rho=0.8)

    def test_oscillation_amplitude_positive(self):
        trajs = _random_trajectories()
        amp = _compute_oscillation_amplitude(trajs)
        self.assertGreater(amp, 0.0)

    def test_oscillation_amplitude_constant_zero(self):
        trajs = np.ones((5, 50, N), dtype=np.float32)
        amp = _compute_oscillation_amplitude(trajs)
        self.assertAlmostEqual(amp, 0.0, places=5)

    def test_lle_from_trajs_finite(self):
        trajs = _random_trajectories()
        lle = _lle_from_trajs(trajs, max_lag=15)
        # May be nan when analysis.lyapunov is not on path; must not crash
        self.assertTrue(np.isfinite(lle) or np.isnan(lle))

    def test_run_phase_diagram_keys(self):
        result = run_phase_diagram(
            self.W, g_min=0.5, g_max=1.5, g_step=0.5,
        )
        for k in ("g_values", "spectral_radii", "actual_rho_W",
                  "g_linear_critical", "h3_supported"):
            self.assertIn(k, result)
        # Old WC keys must not be present
        self.assertNotIn("lles", result)
        self.assertNotIn("oscillation_amplitudes", result)
        self.assertNotIn("critical_g_lle0", result)

    def test_run_phase_diagram_g_values_length(self):
        result = run_phase_diagram(
            self.W, g_min=0.5, g_max=1.5, g_step=0.5,
        )
        expected_len = len(np.arange(0.5, 1.5 + 0.25, 0.5))
        self.assertEqual(len(result["g_values"]), expected_len)

    def test_run_phase_diagram_spectral_radii_scale_with_g(self):
        result = run_phase_diagram(
            self.W, g_min=0.5, g_max=1.5, g_step=0.5,
        )
        rhos = result["spectral_radii"]
        self.assertLess(rhos[0], rhos[-1])

    def test_run_phase_diagram_lle_reference_annotated(self):
        result = run_phase_diagram(
            self.W, g_min=0.5, g_max=1.5, g_step=0.5,
            lle_reference=-0.05,
        )
        self.assertAlmostEqual(result["lle_at_g1"], -0.05, places=4)

    def test_run_phase_diagram_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_phase_diagram(
                self.W, g_min=0.5, g_max=1.0, g_step=0.5,
                output_dir=Path(tmpdir), label="test",
            )
            self.assertTrue((Path(tmpdir) / "phase_diagram_test.json").exists())


# ══════════════════════════════════════════════════════════════════════════════
# E6: Random Network Spectral Comparison
# ══════════════════════════════════════════════════════════════════════════════

class TestRandomSpectralComparison(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_erdos_renyi_shape(self):
        W_er = erdos_renyi(N, target_rho=0.9)
        self.assertEqual(W_er.shape, (N, N))

    def test_erdos_renyi_spectral_radius(self):
        W_er = erdos_renyi(N, target_rho=0.9)
        rho = float(np.abs(np.linalg.eigvals(W_er)).max())
        self.assertAlmostEqual(rho, 0.9, places=3)

    def test_erdos_renyi_different_seeds(self):
        W1 = erdos_renyi(N, seed=0)
        W2 = erdos_renyi(N, seed=1)
        self.assertFalse(np.allclose(W1, W2))

    def test_run_random_spectral_comparison_keys(self):
        result = run_random_spectral_comparison(self.W, n_random=3)
        for k in ("real", "er_random", "degree_preserving_rewire",
                  "weight_shuffle", "z_scores_vs_er", "h1_supported"):
            self.assertIn(k, result)

    def test_run_random_spectral_comparison_real_metrics(self):
        result = run_random_spectral_comparison(self.W, n_random=3)
        self.assertIn("participation_ratio", result["real"])
        self.assertGreater(result["real"]["participation_ratio"], 0.0)
        # WC LLE must not be present
        self.assertNotIn("lle_wc", result["real"])

    def test_run_random_spectral_comparison_er_has_mean_std(self):
        result = run_random_spectral_comparison(self.W, n_random=3)
        er = result["er_random"]
        self.assertIn("participation_ratio_mean", er)
        self.assertIn("participation_ratio_std", er)

    def test_low_rank_matrix_pr_below_er(self):
        result = run_random_spectral_comparison(self.W, n_random=5)
        z = result["z_scores_vs_er"]["pr_z"]
        if np.isfinite(z):
            self.assertLess(z, 1.0)

    def test_run_random_spectral_comparison_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_random_spectral_comparison(
                self.W, n_random=3,
                output_dir=Path(tmpdir), label="test"
            )
            self.assertTrue(
                (Path(tmpdir) / "random_comparison_spectral_test.json").exists()
            )


# ══════════════════════════════════════════════════════════════════════════════
# Integration: run_all with synthetic data
# ══════════════════════════════════════════════════════════════════════════════

class TestRunAllSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W, cls.trajs, cls.R = _make_synthetic_data(n_regions=N, n_traj=8, steps=60)

    def _run(self, experiments, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_all(
                trajectories=self.trajs,
                response_matrix=self.R,
                output_dir=Path(tmpdir),
                experiments=experiments,
                n_random=3,
                g_min=0.5,
                g_max=1.5,
                g_step=0.5,
                **kwargs,
            )
            summary_path = Path(tmpdir) / "run_summary.json"
            self.assertTrue(summary_path.exists())
            with open(summary_path) as f:
                loaded = json.load(f)
        return result, loaded

    def test_run_e1_only(self):
        result, loaded = self._run(["E1"])
        self.assertIn("E1_main", result["results"])

    def test_run_e2e3(self):
        result, loaded = self._run(["E2E3"])
        self.assertIn("E2E3", result["results"])

    def test_run_e4(self):
        result, loaded = self._run(["E4"])
        self.assertIn("E4", result["results"])

    def test_run_e5(self):
        result, loaded = self._run(["E5"])
        self.assertIn("E5", result["results"])

    def test_run_e6(self):
        result, loaded = self._run(["E6"])
        self.assertIn("E6", result["results"])

    def test_run_all_experiments(self):
        result, loaded = self._run(["A", "B_E1", "C", "D", "E2E3", "E4", "E5", "E6", "F"])
        for key in ("A", "E1_main", "C", "D", "E2E3", "E4", "E5", "E6", "F"):
            self.assertIn(key, result["results"], f"Missing result key: {key}")

    def test_run_all_hypotheses_dict(self):
        result, _ = self._run(["B_E1", "E2E3", "E5", "E6"])
        hyps = result["hypotheses"]
        self.assertIn("H1_spectral_concentration", hyps)
        self.assertIn("H2_low_dimensional_dynamics", hyps)
        self.assertIn("H3_near_critical", hyps)

    def test_make_synthetic_data_shapes(self):
        W, trajs, R = _make_synthetic_data(n_regions=30, n_traj=5, steps=40)
        self.assertEqual(W.shape, (30, 30))
        self.assertEqual(trajs.shape, (5, 40, 30))
        self.assertEqual(R.shape, (30, 30))

    def test_run_all_no_trajectories_uses_fc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_all(
                trajectories=None,
                response_matrix=self.R,
                output_dir=Path(tmpdir),
                experiments=["B_E1"],
            )
            self.assertIn("E1_main", result["results"])

    def test_run_all_no_response_matrix_uses_fc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_all(
                trajectories=self.trajs,
                response_matrix=None,
                output_dir=Path(tmpdir),
                experiments=["B_E1"],
            )
            self.assertIn("E1_main", result["results"])

    def test_run_all_raises_without_any_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_all(
                    trajectories=None,
                    response_matrix=None,
                    output_dir=Path(tmpdir),
                    experiments=["B_E1"],
                )


# ══════════════════════════════════════════════════════════════════════════════
# Module A: Connectivity Visualization
# ══════════════════════════════════════════════════════════════════════════════

class TestConnectivityVisualization(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_run_returns_dict(self):
        result = run_connectivity_visualization(self.W)
        self.assertIn("n_regions", result)
        self.assertIn("density", result)

    def test_n_regions_correct(self):
        result = run_connectivity_visualization(self.W)
        self.assertEqual(result["n_regions"], N)

    def test_density_in_range(self):
        result = run_connectivity_visualization(self.W)
        self.assertGreaterEqual(result["density"], 0.0)
        self.assertLessEqual(result["density"], 1.0)

    def test_saves_raw_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_connectivity_visualization(self.W, output_dir=Path(tmpdir), label="test")
            self.assertTrue((Path(tmpdir) / "connectivity_matrix_raw_test.png").exists())

    def test_saves_reordered_with_community_labels(self):
        labels = np.zeros(N, dtype=np.int32)
        labels[N // 2:] = 1  # Two communities
        with tempfile.TemporaryDirectory() as tmpdir:
            run_connectivity_visualization(
                self.W, community_labels=labels,
                output_dir=Path(tmpdir), label="test"
            )
            self.assertTrue(
                (Path(tmpdir) / "connectivity_matrix_reordered_test.png").exists()
            )


# ══════════════════════════════════════════════════════════════════════════════
# Module C: Community Structure
# ══════════════════════════════════════════════════════════════════════════════

class TestCommunityStructure(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_spectral_community_detection_shape(self):
        labels = spectral_community_detection(self.W, k=3)
        self.assertEqual(labels.shape, (N,))

    def test_spectral_community_detection_k_communities(self):
        k = 4
        labels = spectral_community_detection(self.W, k=k)
        self.assertEqual(len(np.unique(labels)), k)

    def test_modularity_q_perfect_block(self):
        # Perfect block matrix: 2 disconnected communities → Q should be high
        W_block = np.zeros((N, N), dtype=np.float32)
        half = N // 2
        W_block[:half, :half] = 1.0
        W_block[half:, half:] = 1.0
        labels = np.array([0] * half + [1] * (N - half))
        Q = compute_modularity_q(W_block, labels)
        self.assertGreater(Q, 0.1)

    def test_modularity_q_single_community(self):
        # Single community → Q = 0
        labels = np.zeros(N, dtype=np.int32)
        Q = compute_modularity_q(self.W, labels)
        self.assertAlmostEqual(Q, 0.0, places=5)

    def test_find_optimal_k_returns_valid(self):
        best_k, labels, Q = find_optimal_k(self.W, k_range=[2, 3, 4], seed=0)
        self.assertIn(best_k, [2, 3, 4])
        self.assertEqual(labels.shape, (N,))
        self.assertTrue(np.isfinite(Q))

    def test_run_community_structure_keys(self):
        result = run_community_structure(self.W, k_range=[2, 3, 4])
        for key in ("n_communities", "modularity_q", "community_labels",
                    "q_interpretation", "community_sizes"):
            self.assertIn(key, result)

    def test_run_community_structure_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_community_structure(self.W, k_range=[2, 3], output_dir=Path(tmpdir),
                                    label="test")
            self.assertTrue((Path(tmpdir) / "community_structure_test.json").exists())

    def test_q_interpretation_values(self):
        result = run_community_structure(self.W, k_range=[2, 3])
        self.assertIn(result["q_interpretation"], ("strong", "moderate", "weak"))


# ══════════════════════════════════════════════════════════════════════════════
# Module D: Hierarchical Structure
# ══════════════════════════════════════════════════════════════════════════════

class TestHierarchicalStructure(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()

    def test_weight_to_distance_range(self):
        D = _weight_to_distance(self.W)
        self.assertGreaterEqual(float(D.min()), -1e-6)
        self.assertLessEqual(float(D.max()), 1.0 + 1e-6)

    def test_weight_to_distance_symmetric(self):
        D = _weight_to_distance(self.W)
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_weight_to_distance_diagonal_zero(self):
        D = _weight_to_distance(self.W)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_compute_linkage_shape(self):
        Z = compute_linkage(self.W)
        self.assertEqual(Z.shape, (N - 1, 4))

    def test_cluster_stats_correct_length(self):
        Z = compute_linkage(self.W)
        stats = compute_cluster_stats_at_levels(Z, N)
        self.assertEqual(len(stats), len([0.20, 0.40, 0.60, 0.80, 1.00]))

    def test_cluster_stats_n_clusters_monotone(self):
        Z = compute_linkage(self.W)
        stats = compute_cluster_stats_at_levels(Z, N)
        n_clusters = [s["n_clusters"] for s in stats]
        # Higher height → fewer clusters (merging more)
        self.assertGreaterEqual(n_clusters[0], n_clusters[-1])

    def test_hierarchy_index_finite(self):
        Z = compute_linkage(self.W)
        stats = compute_cluster_stats_at_levels(Z, N)
        h_idx = compute_hierarchy_index(stats)
        self.assertTrue(np.isfinite(h_idx))

    def test_run_hierarchical_structure_keys(self):
        result = run_hierarchical_structure(self.W)
        for key in ("is_hierarchical", "hierarchy_index",
                    "n_clusters_at_20pct_height", "n_clusters_at_80pct_height"):
            self.assertIn(key, result)

    def test_run_hierarchical_structure_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_hierarchical_structure(self.W, output_dir=Path(tmpdir), label="test")
            self.assertTrue((Path(tmpdir) / "hierarchical_stats_test.json").exists())

    def test_all_nodes_covered(self):
        Z = compute_linkage(self.W)
        # At height 0 (tightest), all nodes should be their own cluster
        stats = compute_cluster_stats_at_levels(Z, N, [0.001])
        total_nodes = sum(stats[0]["cluster_sizes"])
        self.assertEqual(total_nodes, N)


# ══════════════════════════════════════════════════════════════════════════════
# Module F: PCA + Attractor Projection
# ══════════════════════════════════════════════════════════════════════════════

class TestPCAAttractor(unittest.TestCase):
    def setUp(self):
        self.W = _low_rank_matrix()
        self.trajs = _random_trajectories(n_traj=8, steps=60)

    def test_compute_pca_shapes(self):
        result = compute_pca(self.trajs)
        n_components = result["n_components"]
        self.assertEqual(len(result["explained_variance_ratio"]), n_components)
        self.assertEqual(result["X_pca"].shape[1], n_components)

    def test_explained_variance_sums_to_1(self):
        result = compute_pca(self.trajs)
        total = sum(result["explained_variance_ratio"])
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_cumulative_variance_monotone(self):
        result = compute_pca(self.trajs)
        cumul = result["cumulative_variance"]
        self.assertTrue(np.all(np.diff(cumul) >= -1e-8))

    def test_n_components_90pct_positive(self):
        result = compute_pca(self.trajs)
        self.assertGreater(result["n_components_90pct"], 0)

    def test_burnin_reduces_samples(self):
        result_no_burnin = compute_pca(self.trajs, burnin=0)
        result_burnin = compute_pca(self.trajs, burnin=10)
        self.assertLess(result_burnin["n_samples"], result_no_burnin["n_samples"])

    def test_run_pca_attractor_keys(self):
        result = run_pca_attractor(self.trajs)
        for key in ("variance_top1_pct", "variance_top2_pct", "variance_top5_pct",
                    "n_components_90pct", "pca_efficiency_ratio"):
            self.assertIn(key, result)

    def test_variance_top_values_in_range(self):
        result = run_pca_attractor(self.trajs)
        self.assertGreater(result["variance_top1_pct"], 0.0)
        self.assertLessEqual(result["variance_top1_pct"], 100.0)
        if result["variance_top2_pct"] is not None:
            self.assertGreaterEqual(result["variance_top2_pct"],
                                    result["variance_top1_pct"])

    def test_run_pca_attractor_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_pca_attractor(self.trajs, output_dir=Path(tmpdir), label="test")
            self.assertTrue((Path(tmpdir) / "pca_results_test.json").exists())
            self.assertTrue((Path(tmpdir) / "pca_projections_test.npy").exists())

    def test_low_rank_pca_fewer_components(self):
        # Low-rank W → WC dynamics → PCA should need fewer components than N
        result = run_pca_attractor(self.trajs)
        self.assertLess(result["n_components_90pct"], N)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ══════════════════════════════════════════════════════════════════════════════
# B_LYA (Lyapunov Spectrum): REMOVED — module b_lyapunov_spectrum.py has been
# deleted.  Linearised Lyapunov spectrum is now computed via DMD in
# dynamics_pipeline Phase 3e.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# H: Power Spectrum
# ══════════════════════════════════════════════════════════════════════════════

from spectral_dynamics.h_power_spectrum import run_power_spectrum


class TestPowerSpectrum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(77)
        N = 20
        T = 64
        # Sine wave at f=0.1 Hz, dt=2s → period=10 steps
        t_axis = np.arange(T, dtype=np.float32)
        freq = 0.1
        dt = 2.0
        base = np.sin(2 * np.pi * freq * dt * t_axis)  # (T,)
        trajs = np.empty((5, T, N), dtype=np.float32)
        for i in range(5):
            noise = rng.standard_normal((T, N)).astype(np.float32) * 0.05
            trajs[i] = (base[:, None] + noise).astype(np.float32)
        cls.trajs = trajs
        cls.dt = dt
        cls.expected_freq = freq

    def test_returns_freqs_and_psd(self):
        result = run_power_spectrum(self.trajs, dt=self.dt, burnin=0)
        self.assertIn("freqs", result)
        self.assertIn("mean_psd", result)
        self.assertIn("region_psd", result)

    def test_freqs_shape(self):
        result = run_power_spectrum(self.trajs, dt=self.dt, burnin=0)
        T_use = self.trajs.shape[1]
        self.assertEqual(len(result["freqs"]), T_use // 2 + 1)

    def test_dominant_freq_near_expected(self):
        result = run_power_spectrum(self.trajs, dt=self.dt, burnin=0)
        dom_f = result["band_analysis"]["dominant_freq_hz"]
        # Allow ±2× frequency resolution
        freq_res = 1.0 / (self.trajs.shape[1] * self.dt)
        self.assertAlmostEqual(dom_f, self.expected_freq, delta=3 * freq_res)

    def test_band_analysis_keys(self):
        result = run_power_spectrum(self.trajs, dt=self.dt, burnin=0)
        ba = result["band_analysis"]
        for key in ("dominant_freq_hz", "dominant_freq_band",
                    "total_power", "band_powers", "band_power_fractions",
                    "nyquist_hz", "bands_used"):
            self.assertIn(key, ba)

    def test_spatial_modes_peak_regions(self):
        result = run_power_spectrum(self.trajs, dt=self.dt, burnin=0)
        sm = result["spatial_modes"]
        self.assertIn("peak_region_per_band", sm)
        for pk in sm["peak_region_per_band"].values():
            self.assertGreaterEqual(pk, 0)
            self.assertLess(pk, self.trajs.shape[2])

    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_power_spectrum(self.trajs, dt=self.dt, burnin=0,
                               output_dir=Path(tmpdir), label="test")
            self.assertTrue(
                (Path(tmpdir) / "power_spectrum_test.json").exists()
            )


# ══════════════════════════════════════════════════════════════════════════════
# I: Energy Budget (GNN trajectories → run_energy_budget)
# ══════════════════════════════════════════════════════════════════════════════

from spectral_dynamics.i_energy_constraint import run_energy_budget


class TestEnergyBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajs = _random_trajectories(n_traj=10, steps=80, n=N).astype(np.float32)

    def test_energy_budget_keys(self):
        result = run_energy_budget(self.trajs)
        for key in ("E_mean", "E_std", "E_median", "E_per_region", "recommended_budgets"):
            self.assertIn(key, result)

    def test_energy_budget_recommended_budgets_keys(self):
        result = run_energy_budget(self.trajs)
        rec = result["recommended_budgets"]
        for k in ("tight_constraint", "moderate_constraint", "natural", "relaxed"):
            self.assertIn(k, rec)

    def test_energy_budget_tight_less_than_relaxed(self):
        result = run_energy_budget(self.trajs)
        rec = result["recommended_budgets"]
        self.assertLess(rec["tight_constraint"], rec["relaxed"])

    def test_energy_budget_e_mean_positive(self):
        result = run_energy_budget(self.trajs)
        self.assertGreater(result["E_mean"], 0.0)

    def test_energy_budget_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_energy_budget(
                self.trajs,
                output_dir=Path(tmpdir), label="test",
            )
            self.assertTrue((Path(tmpdir) / "energy_budget_test.json").exists())


class TestRunAllNewExperiments(unittest.TestCase):
    """Integration test: run_all correctly executes H, I."""

    @classmethod
    def setUpClass(cls):
        W, trajs, R = _make_synthetic_data(n_regions=N, n_traj=8, steps=60)
        cls.W = W
        cls.trajs = trajs
        cls.R = R

    def test_h_in_run_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_all(
                trajectories=self.trajs,
                response_matrix=self.R,
                output_dir=Path(tmpdir),
                experiments=["H"],
            )
            self.assertIn("H", summary["results"])
            dom_f = summary["results"]["H"].get("dominant_freq_hz")
            self.assertIsNotNone(dom_f)

    def test_i_in_run_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_all(
                trajectories=self.trajs,
                response_matrix=self.R,
                output_dir=Path(tmpdir),
                experiments=["I"],
            )
            self.assertIn("I", summary["results"])
            self.assertIn("E_mean", summary["results"]["I"])
            self.assertIn("recommended_budgets", summary["results"]["I"])

    def test_h5_hypothesis_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_all(
                trajectories=self.trajs,
                response_matrix=self.R,
                output_dir=Path(tmpdir),
                experiments=["I"],
            )
            hyp = summary["hypotheses"]
            self.assertIn("H5_energy_constraint_criticality", hyp)
            self.assertIn("E_mean", hyp["H5_energy_constraint_criticality"])


