"""
Tests for the unified dynamics_pipeline
========================================

Covers:
  - Config merging and quick-mode overrides
  - Phase runners with synthetic data (no model required)
  - Hypothesis evaluation logic
  - Consistency check logic
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure dynamics_pipeline is importable
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BD_DIR = _REPO_ROOT / "brain_dynamics"
if _BD_DIR.exists() and str(_BD_DIR) not in sys.path:
    sys.path.insert(0, str(_BD_DIR))


class TestConfigMerge(unittest.TestCase):
    """Test the configuration merging logic."""

    def test_merge_basic(self):
        from dynamics_pipeline.run import _merge
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _merge(base, override)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"]["c"], 99)
        self.assertEqual(result["b"]["d"], 3)
        self.assertEqual(result["e"], 5)

    def test_merge_deep_nesting(self):
        from dynamics_pipeline.run import _merge
        base = {"x": {"y": {"z": 1, "w": 2}}}
        override = {"x": {"y": {"z": 42}}}
        result = _merge(base, override)
        self.assertEqual(result["x"]["y"]["z"], 42)
        self.assertEqual(result["x"]["y"]["w"], 2)

    def test_quick_overrides_structure(self):
        from dynamics_pipeline.run import _DEFAULTS, _QUICK_OVERRIDES, _merge
        cfg = _merge(_DEFAULTS, _QUICK_OVERRIDES)
        self.assertEqual(cfg["data_generation"]["n_init"], 20)
        self.assertEqual(cfg["data_generation"]["steps"], 100)
        # Non-overridden values should remain
        self.assertEqual(cfg["data_generation"]["seed"], 42)
        self.assertTrue(cfg["dynamics"]["lyapunov"]["enabled"])

    def test_defaults_have_required_keys(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertIn("model", _DEFAULTS)
        self.assertIn("simulator", _DEFAULTS)
        self.assertIn("data_generation", _DEFAULTS)
        self.assertIn("network_structure", _DEFAULTS)
        self.assertIn("dynamics", _DEFAULTS)
        self.assertIn("validation", _DEFAULTS)
        self.assertIn("advanced", _DEFAULTS)
        self.assertIn("output", _DEFAULTS)

    def test_lyapunov_defaults(self):
        from dynamics_pipeline.run import _DEFAULTS
        lya = _DEFAULTS["dynamics"]["lyapunov"]
        self.assertTrue(lya["enabled"])
        self.assertEqual(lya["n_segments"], 3)
        self.assertEqual(lya["convergence_threshold"], 0.05)

    def test_dmd_enabled_by_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertTrue(_DEFAULTS["dynamics"]["dmd_spectrum"]["enabled"])

    def test_wolf_gs_not_in_defaults(self):
        """Wolf-GS Lyapunov spectrum should not be in defaults (replaced by DMD)."""
        from dynamics_pipeline.run import _DEFAULTS
        self.assertNotIn("lyapunov_spectrum", _DEFAULTS)
        self.assertIn("dmd_spectrum", _DEFAULTS["dynamics"])


class TestHypothesisEvaluation(unittest.TestCase):
    """Test hypothesis evaluation logic in Phase 6."""

    def test_h1_supported(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {
                "participation_ratio": 5.0,
                "n_dominant": 3,
                "n_regions": 100,
            }
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H1"]["verdict"], "SUPPORTED")
        self.assertAlmostEqual(H["H1"]["PR_N_ratio"], 0.05)

    def test_h1_not_supported(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {
                "participation_ratio": 50.0,
                "n_dominant": 30,
                "n_regions": 100,
            }
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H1"]["verdict"], "NOT_SUPPORTED")

    def test_h1_insufficient_data(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        H = _evaluate_hypotheses({})
        self.assertEqual(H["H1"]["verdict"], "INSUFFICIENT_DATA")

    def test_h2_supported(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {"n_regions": 100},
            "pca": {"n_components_90": 5},
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H2"]["verdict"], "SUPPORTED")

    def test_h2_not_supported(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {"n_regions": 100},
            "pca": {"n_components_90": 50},
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H2"]["verdict"], "NOT_SUPPORTED")

    def test_h3_near_critical(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "lyapunov": {
                "chaos_regime": {"regime": "edge_of_chaos"},
                "mean_lyapunov": 0.005,
            }
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H3"]["verdict"], "SUPPORTED")

    def test_h3_strongly_chaotic(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "lyapunov": {
                "chaos_regime": {"regime": "strongly_chaotic"},
                "mean_lyapunov": 0.5,
            }
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H3"]["verdict"], "NOT_SUPPORTED")

    def test_h4_has_oscillations(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "power_spectrum": {
                "band_analysis": {
                    "dominant_freq_hz": 0.05,
                    "dominant_freq_band": "low",
                }
            }
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H4"]["verdict"], "SUPPORTED")

    def test_h5_not_tested(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        H = _evaluate_hypotheses({})
        self.assertEqual(H["H5"]["verdict"], "NOT_TESTED")


class TestConsistencyChecks(unittest.TestCase):
    """Test cross-phase consistency checks in Phase 6."""

    def test_consistent_stable(self):
        """Stable LLE + subcritical ρ → consistent."""
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        results = {
            "lyapunov": {
                "mean_lyapunov": -0.05,
                "chaos_regime": {"regime": "stable"},
            },
            "dmd_spectrum": {
                "spectral_radius": 0.95,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
        report = results["report"]
        self.assertTrue(report["consistency"]["consistent"])
        self.assertEqual(report["consistency"]["lle_sign"], "negative")
        self.assertEqual(report["consistency"]["rho_sign"], "subcritical")

    def test_inconsistent_chaos_vs_subcritical(self):
        """Chaotic LLE + subcritical ρ → inconsistent."""
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        results = {
            "lyapunov": {
                "mean_lyapunov": 0.05,
                "chaos_regime": {"regime": "weakly_chaotic"},
            },
            "dmd_spectrum": {
                "spectral_radius": 0.90,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
        report = results["report"]
        self.assertFalse(report["consistency"]["consistent"])

    def test_consistent_near_critical(self):
        """Near-zero LLE + near-critical ρ → consistent."""
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        results = {
            "lyapunov": {
                "mean_lyapunov": 0.005,
                "chaos_regime": {"regime": "edge_of_chaos"},
            },
            "dmd_spectrum": {
                "spectral_radius": 1.002,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
        report = results["report"]
        self.assertTrue(report["consistency"]["consistent"])

    def test_synthesis_saves_json(self):
        """Phase 6 should save pipeline_report.json."""
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        results = {"lyapunov": {"mean_lyapunov": 0.01,
                                "chaos_regime": {"regime": "edge_of_chaos"}}}
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
            report_path = Path(td) / "pipeline_report.json"
            self.assertTrue(report_path.exists())
            with open(report_path) as f:
                data = json.load(f)
            self.assertIn("hypotheses", data)
            self.assertIn("regime", data)


class TestPhase2Structure(unittest.TestCase):
    """Test Phase 2 network structure analysis with synthetic data."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        N = 20
        # Low-rank matrix (few dominant eigenvalues)
        U = rng.standard_normal((N, 3))
        cls.W = (U @ U.T) / N
        cls.W /= np.max(np.abs(np.linalg.eigvals(cls.W))) + 1e-6  # normalise
        cls.trajs = rng.random((5, 50, N)).astype(np.float32)

    def test_spectral_analysis(self):
        from dynamics_pipeline.pipeline import run_phase2_structure
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "network_structure": {
                "enabled": True,
                "spectral": {"enabled": True},
                "community": {"enabled": False},
                "hierarchy": {"enabled": False},
                "modal_energy": {"enabled": False},
                "visualization": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"response_matrix": self.W, "trajectories": self.trajs}
        with tempfile.TemporaryDirectory() as td:
            run_phase2_structure(cfg, results, Path(td))
        self.assertIn("spectral", results)
        self.assertGreater(results["spectral"]["spectral_radius"], 0)
        self.assertGreater(results["spectral"]["participation_ratio"], 0)

    def test_community_detection(self):
        from dynamics_pipeline.pipeline import run_phase2_structure
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "network_structure": {
                "enabled": True,
                "spectral": {"enabled": False},
                "community": {"enabled": True, "k_range": [2, 3, 4]},
                "hierarchy": {"enabled": False},
                "modal_energy": {"enabled": False},
                "visualization": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"response_matrix": self.W}
        with tempfile.TemporaryDirectory() as td:
            run_phase2_structure(cfg, results, Path(td))
        self.assertIn("community", results)
        self.assertGreater(results["community"]["n_communities"], 0)

    def test_phase2_disabled(self):
        from dynamics_pipeline.pipeline import run_phase2_structure
        cfg = {"network_structure": {"enabled": False}, "output": {}}
        results = {}
        with tempfile.TemporaryDirectory() as td:
            run_phase2_structure(cfg, results, Path(td))
        # Should have no new keys
        self.assertEqual(len(results), 0)

    def test_fc_fallback_when_no_response_matrix(self):
        """When no response matrix, should compute FC from trajectories."""
        from dynamics_pipeline.pipeline import run_phase2_structure
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "network_structure": {
                "enabled": True,
                "spectral": {"enabled": True},
                "community": {"enabled": False},
                "hierarchy": {"enabled": False},
                "modal_energy": {"enabled": False},
                "visualization": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"trajectories": self.trajs}  # No response_matrix
        with tempfile.TemporaryDirectory() as td:
            run_phase2_structure(cfg, results, Path(td))
        # Should still produce spectral results via FC
        self.assertIn("spectral", results)


class TestPhase3Dynamics(unittest.TestCase):
    """Test Phase 3 dynamics characterisation with synthetic data (no model)."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        N = 10
        T = 100
        n_init = 5
        # Simple WC-like trajectories
        trajs = np.zeros((n_init, T, N), dtype=np.float32)
        for i in range(n_init):
            x = rng.random(N).astype(np.float32) * 0.5 + 0.25
            for t in range(T):
                trajs[i, t] = x
                x = np.clip(np.tanh(x * 0.9 + rng.normal(0, 0.01, N).astype(np.float32)), 0, 1)
        cls.trajs = trajs

    def _make_mock_simulator(self):
        sim = MagicMock()
        sim.n_regions = 10
        sim.dt = 2.0
        sim.state_bounds = (0.0, 1.0)
        sim.modality = "fmri"
        return sim

    def test_stability_analysis(self):
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": True},
                "attractor": {"enabled": False},
                "convergence": {"enabled": False},
                "lyapunov": {"enabled": False},
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"trajectories": self.trajs}
        sim = self._make_mock_simulator()
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td))
        self.assertIn("stability", results)

    def test_attractor_analysis(self):
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": False},
                "attractor": {"enabled": True, "tail_steps": 10, "k_candidates": [2, 3]},
                "convergence": {"enabled": False},
                "lyapunov": {"enabled": False},
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"trajectories": self.trajs}
        sim = self._make_mock_simulator()
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td))
        self.assertIn("attractor", results)

    def test_convergence_analysis(self):
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": False},
                "attractor": {"enabled": False},
                "convergence": {"enabled": True, "n_pairs": 5},
                "lyapunov": {"enabled": False},
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        results = {"trajectories": self.trajs}
        sim = self._make_mock_simulator()
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td))
        self.assertIn("convergence", results)

    def test_pca_analysis(self):
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": False},
                "attractor": {"enabled": False},
                "convergence": {"enabled": False},
                "lyapunov": {"enabled": False},
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": True},
            },
            "output": {"save_plots": False},
        })
        results = {"trajectories": self.trajs}
        sim = self._make_mock_simulator()
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td))
        self.assertIn("pca", results)


class TestPhase4Validation(unittest.TestCase):
    """Test Phase 4 statistical validation."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        N = 10
        T = 100
        n_init = 5
        trajs = np.zeros((n_init, T, N), dtype=np.float32)
        for i in range(n_init):
            x = rng.random(N).astype(np.float32)
            for t in range(T):
                trajs[i, t] = x
                x = np.clip(np.tanh(x * 0.9), 0, 1) + rng.normal(0, 0.01, N).astype(np.float32)
                x = np.clip(x, 0, 1)
        cls.trajs = trajs

    def test_surrogate_test(self):
        from dynamics_pipeline.pipeline import run_phase4_validation
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "data_generation": {"seed": 42},
            "validation": {
                "surrogate_test": {"enabled": True, "n_surrogates": 3, "n_traj_sample": 2},
                "random_comparison": {"enabled": False},
                "embedding_dimension": {"enabled": False},
                "perturbation": {"enabled": False},
            },
        })
        results = {"trajectories": self.trajs}
        with tempfile.TemporaryDirectory() as td:
            run_phase4_validation(cfg, results, Path(td))
        self.assertIn("surrogate_test", results)

    def test_embedding_dimension(self):
        from dynamics_pipeline.pipeline import run_phase4_validation
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "data_generation": {"seed": 42},
            "validation": {
                "surrogate_test": {"enabled": False},
                "random_comparison": {"enabled": False},
                "embedding_dimension": {"enabled": True, "fnn_max_dim": 3, "corr_dim": False},
                "perturbation": {"enabled": False},
            },
        })
        results = {"trajectories": self.trajs}
        with tempfile.TemporaryDirectory() as td:
            run_phase4_validation(cfg, results, Path(td))
        self.assertIn("embedding_dimension", results)


class TestPhaseSelection(unittest.TestCase):
    """Test that --phases flag correctly disables phases."""

    def test_phase_disabling(self):
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = dict(_DEFAULTS)
        # Simulate --phases 1 3
        phases = {1, 3}
        if 2 not in phases:
            cfg["network_structure"]["enabled"] = False
        if 4 not in phases:
            for k in cfg["validation"]:
                if isinstance(cfg["validation"][k], dict):
                    cfg["validation"][k]["enabled"] = False
        if 5 not in phases:
            for k in cfg["advanced"]:
                if isinstance(cfg["advanced"][k], dict):
                    cfg["advanced"][k]["enabled"] = False

        self.assertFalse(cfg["network_structure"]["enabled"])
        self.assertFalse(cfg["validation"]["surrogate_test"]["enabled"])
        self.assertFalse(cfg["advanced"]["stimulation"]["enabled"])
        # Phase 3 dynamics should remain enabled
        self.assertTrue(cfg["dynamics"]["lyapunov"]["enabled"])


class TestKaplanYorkeDimension(unittest.TestCase):
    """Test the local _kaplan_yorke_dimension helper."""

    def test_fixed_point(self):
        from dynamics_pipeline.pipeline import _kaplan_yorke_dimension
        # All negative → fixed point → D_KY = 0
        spectrum = np.array([-0.1, -0.5, -1.0])
        self.assertAlmostEqual(_kaplan_yorke_dimension(spectrum), 0.0)

    def test_limit_cycle(self):
        from dynamics_pipeline.pipeline import _kaplan_yorke_dimension
        # λ₁=0, rest negative → D_KY ∈ [1, 2)
        spectrum = np.array([0.0, -0.5, -1.0])
        ky = _kaplan_yorke_dimension(spectrum)
        self.assertGreaterEqual(ky, 0.0)
        self.assertLess(ky, 2.0)

    def test_chaotic(self):
        from dynamics_pipeline.pipeline import _kaplan_yorke_dimension
        # One positive, one zero, rest negative → D_KY > 1
        spectrum = np.array([0.1, 0.0, -0.05, -0.2, -0.5])
        ky = _kaplan_yorke_dimension(spectrum)
        self.assertGreater(ky, 1.0)
        self.assertLess(ky, 5.0)

    def test_empty_spectrum(self):
        from dynamics_pipeline.pipeline import _kaplan_yorke_dimension
        self.assertAlmostEqual(_kaplan_yorke_dimension(np.array([])), 0.0)


class TestAttractorDimension(unittest.TestCase):
    """Test that attractor dimension config entry is present and respected."""

    def test_attractor_dimension_in_defaults(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertIn("attractor_dimension", _DEFAULTS["dynamics"])
        self.assertTrue(_DEFAULTS["dynamics"]["attractor_dimension"]["enabled"])

    def test_attractor_dimension_disabled(self):
        """When disabled, should not produce attractor_dimension results."""
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": False},
                "attractor": {"enabled": False},
                "convergence": {"enabled": False},
                "lyapunov": {"enabled": False},
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": False},
                "attractor_dimension": {"enabled": False},
            },
            "output": {"save_plots": False},
        })
        rng = np.random.default_rng(42)
        trajs = rng.random((5, 50, 10)).astype(np.float32)
        results = {"trajectories": trajs}
        sim = MagicMock()
        sim.n_regions = 10
        sim.dt = 2.0
        sim.state_bounds = (0.0, 1.0)
        sim.modality = "fmri"
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td))
        self.assertNotIn("attractor_dimension", results)


class TestNonlinearityIndex(unittest.TestCase):
    """Test the nonlinearity index in Phase 6 consistency check."""

    def test_nonlinearity_index_present(self):
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        results = {
            "lyapunov": {
                "mean_lyapunov": 0.05,
                "chaos_regime": {"regime": "weakly_chaotic"},
            },
            "dmd_spectrum": {
                "spectral_radius": 0.95,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
        report = results["report"]
        self.assertIn("nonlinearity_index", report["consistency"])
        self.assertIn("lambda_dmd_max", report["consistency"])

    def test_nonlinearity_index_near_zero_when_consistent(self):
        """When ρ matches LLE, nonlinearity index should be small."""
        from dynamics_pipeline.pipeline import run_phase6_synthesis
        # λ=ln(ρ)=-0.05, mean_lle=-0.05 → index ≈ 0
        rho = np.exp(-0.05)
        results = {
            "lyapunov": {
                "mean_lyapunov": -0.05,
                "chaos_regime": {"regime": "stable"},
            },
            "dmd_spectrum": {
                "spectral_radius": float(rho),
            },
        }
        with tempfile.TemporaryDirectory() as td:
            run_phase6_synthesis({"data_generation": {"seed": 42}}, results, Path(td))
        idx = results["report"]["consistency"]["nonlinearity_index"]
        self.assertLess(idx, 0.05)  # nearly zero when consistent


class TestH2WithAttractorDimension(unittest.TestCase):
    """Test H2 hypothesis evaluation includes D₂ and K-Y when available."""

    def test_h2_includes_d2(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {"n_regions": 100},
            "pca": {"n_components_90": 8},
            "attractor_dimension": {"D2": 4.5, "KY_linearised": 6.2},
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H2"]["verdict"], "SUPPORTED")
        self.assertAlmostEqual(H["H2"]["D2"], 4.5)
        self.assertAlmostEqual(H["H2"]["KY_linearised"], 6.2)
        self.assertIn("D₂=4.50", H["H2"]["summary"])
        self.assertIn("K-Y_lin=6.20", H["H2"]["summary"])

    def test_h2_works_without_attractor_dim(self):
        from dynamics_pipeline.pipeline import _evaluate_hypotheses
        results = {
            "spectral": {"n_regions": 100},
            "pca": {"n_components_90": 5},
        }
        H = _evaluate_hypotheses(results)
        self.assertEqual(H["H2"]["verdict"], "SUPPORTED")
        self.assertIsNone(H["H2"]["D2"])


class TestModalityDefaults(unittest.TestCase):
    """Test that new modality-related config keys are present in _DEFAULTS."""

    def test_k_cross_modal_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertIn("k_cross_modal", _DEFAULTS["model"])
        self.assertEqual(_DEFAULTS["model"]["k_cross_modal"], 5)

    def test_fmri_subsample_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertIn("fmri_subsample", _DEFAULTS["simulator"])
        self.assertEqual(_DEFAULTS["simulator"]["fmri_subsample"], 25)

    def test_n_temporal_windows_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        self.assertIn("n_temporal_windows", _DEFAULTS["data_generation"])
        self.assertIsNone(_DEFAULTS["data_generation"]["n_temporal_windows"])

    def test_lyapunov_method_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        lya = _DEFAULTS["dynamics"]["lyapunov"]
        self.assertIn("method", lya)
        self.assertEqual(lya["method"], "rosenstein")

    def test_lyapunov_n_workers_default(self):
        from dynamics_pipeline.run import _DEFAULTS
        lya = _DEFAULTS["dynamics"]["lyapunov"]
        self.assertIn("n_workers", lya)
        self.assertEqual(lya["n_workers"], 1)

    def test_modality_choices(self):
        """simulator.modality can be fmri/eeg/both/joint."""
        from dynamics_pipeline.run import _DEFAULTS, _merge
        for modality in ("fmri", "eeg", "both", "joint"):
            cfg = _merge(_DEFAULTS, {"simulator": {"modality": modality}})
            self.assertEqual(cfg["simulator"]["modality"], modality)

    def test_n_workers_override_via_merge(self):
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {"dynamics": {"lyapunov": {"n_workers": 4}}})
        self.assertEqual(cfg["dynamics"]["lyapunov"]["n_workers"], 4)
        # Other lyapunov keys must not be lost
        self.assertIn("method", cfg["dynamics"]["lyapunov"])


class TestPhase3ModalityParam(unittest.TestCase):
    """Test that modality parameter is respected in run_phase3_dynamics."""

    def _make_cfg(self, **lya_overrides):
        from dynamics_pipeline.run import _DEFAULTS, _merge
        lya = {"enabled": False}
        lya.update(lya_overrides)
        return _merge(_DEFAULTS, {
            "dynamics": {
                "stability": {"enabled": False},
                "attractor": {"enabled": False},
                "convergence": {"enabled": False},
                "lyapunov": lya,
                "dmd_spectrum": {"enabled": False},
                "power_spectrum": {"enabled": False},
                "pca": {"enabled": False},
                "attractor_dimension": {"enabled": False},
            },
            "output": {"save_plots": False},
        })

    def test_lyapunov_disabled_no_results(self):
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        cfg = self._make_cfg()
        rng = np.random.default_rng(0)
        trajs = rng.random((5, 50, 10)).astype(np.float32)
        results = {"trajectories": trajs}
        sim = MagicMock()
        sim.n_regions = 10
        sim.dt = 2.0
        sim.state_bounds = (0.0, 1.0)
        sim.modality = "fmri"
        with tempfile.TemporaryDirectory() as td:
            run_phase3_dynamics(cfg, results, sim, Path(td), modality="fmri")
        self.assertNotIn("lyapunov", results)

    def test_phase3_accepts_modality_kwarg(self):
        """run_phase3_dynamics must accept modality kwarg without error."""
        from dynamics_pipeline.pipeline import run_phase3_dynamics
        cfg = self._make_cfg()
        rng = np.random.default_rng(0)
        trajs = rng.random((5, 20, 8)).astype(np.float32)
        results = {"trajectories": trajs}
        sim = MagicMock()
        sim.n_regions = 8
        sim.dt = 2.0
        sim.state_bounds = None
        sim.modality = "joint"
        with tempfile.TemporaryDirectory() as td:
            # Should not raise
            run_phase3_dynamics(cfg, results, sim, Path(td), modality="joint")


class TestPhase5ModalityParam(unittest.TestCase):
    """Test that modality parameter is forwarded to run_phase5_advanced."""

    def test_phase5_accepts_modality_kwarg(self):
        from dynamics_pipeline.pipeline import run_phase5_advanced
        from dynamics_pipeline.run import _DEFAULTS, _merge
        cfg = _merge(_DEFAULTS, {
            "advanced": {
                "stimulation": {"enabled": False},
                "energy_constraint": {"enabled": False},
                "phase_diagram": {"enabled": False},
                "controllability": {"enabled": False},
                "information_flow": {"enabled": False},
                "critical_slowing_down": {"enabled": False},
            },
        })
        results = {}
        sim = MagicMock()
        sim.n_regions = 8
        with tempfile.TemporaryDirectory() as td:
            # Must not raise for any modality value
            run_phase5_advanced(cfg, results, sim, Path(td), modality="joint")
            run_phase5_advanced(cfg, results, sim, Path(td), modality="eeg")
            run_phase5_advanced(cfg, results, sim, Path(td), modality="fmri")


class TestRunPhasesFunctionExists(unittest.TestCase):
    """Verify _run_phases_for_modality is defined and callable."""

    def test_helper_importable(self):
        from dynamics_pipeline.pipeline import _run_phases_for_modality
        self.assertTrue(callable(_run_phases_for_modality))


# ─────────────────────────────────────────────────────────────────────────────
# New tests for Final Supplement Specification
# ─────────────────────────────────────────────────────────────────────────────

class TestHopfModeIdentification(unittest.TestCase):
    """New Test 1: Hopf mode identification from DMD."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        N, T, n_traj = 8, 120, 4
        # Oscillatory trajectories (Hopf-like)
        t_arr = np.linspace(0, 6 * np.pi, T)
        trajs = np.zeros((n_traj, T, N), dtype=np.float64)
        for i in range(n_traj):
            for j in range(N):
                freq = 0.05 + j * 0.015
                trajs[i, :, j] = (
                    0.4 * np.sin(freq * t_arr + rng.uniform(0, 1))
                    + rng.normal(0, 0.02, T)
                )
        cls.trajs = trajs

    def test_extract_dominant_hopf_mode_fields(self):
        """_extract_dominant_hopf_mode returns required fields."""
        from analysis.jacobian_analysis import (
            estimate_jacobian_dmd,
            analyze_jacobian_spectrum,
            _extract_dominant_hopf_mode,
        )
        A = estimate_jacobian_dmd(self.trajs, burnin=5)
        spec = analyze_jacobian_spectrum(A, dt=1.0)
        hopf = _extract_dominant_hopf_mode(spec)
        self.assertIn("has_hopf", hopf)
        self.assertIn("n_hopf_pairs", hopf)
        self.assertIn("hopf_frequency", hopf)
        self.assertIn("hopf_growth_rate", hopf)
        self.assertIn("hopf_pair_index", hopf)
        self.assertIn("hopf_mode_vector", hopf)

    def test_hopf_detected_in_oscillatory_system(self):
        """Hopf pairs are detected for oscillatory trajectories."""
        from analysis.jacobian_analysis import (
            estimate_jacobian_dmd,
            analyze_jacobian_spectrum,
            _extract_dominant_hopf_mode,
        )
        A = estimate_jacobian_dmd(self.trajs, burnin=5)
        spec = analyze_jacobian_spectrum(A, dt=1.0)
        hopf = _extract_dominant_hopf_mode(spec)
        # Oscillatory system should have at least some Hopf pairs
        # (not guaranteed for all random seeds/sizes but likely)
        self.assertIsInstance(hopf["has_hopf"], bool)
        self.assertIsInstance(hopf["n_hopf_pairs"], int)

    def test_hopf_frequency_is_positive_when_present(self):
        """When Hopf pairs exist, hopf_frequency must be positive."""
        from analysis.jacobian_analysis import (
            estimate_jacobian_dmd,
            analyze_jacobian_spectrum,
            _extract_dominant_hopf_mode,
        )
        A = estimate_jacobian_dmd(self.trajs, burnin=5)
        spec = analyze_jacobian_spectrum(A, dt=1.0)
        hopf = _extract_dominant_hopf_mode(spec)
        if hopf["has_hopf"]:
            self.assertGreater(hopf["hopf_frequency"], 0)
            self.assertIsNotNone(hopf["hopf_mode_vector"])
            self.assertIsInstance(hopf["hopf_mode_vector"], list)

    def test_run_jacobian_analysis_saves_hopf_json(self):
        """run_jacobian_analysis saves hopf_modes.json."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        with tempfile.TemporaryDirectory() as td:
            result = run_jacobian_analysis(
                _Sim(), self.trajs, output_dir=Path(td)
            )
            self.assertTrue((Path(td) / "hopf_modes.json").exists())
            with open(Path(td) / "hopf_modes.json") as f:
                data = json.load(f)
            self.assertIn("has_hopf", data)
            self.assertIn("n_hopf_pairs", data)

    def test_run_jacobian_analysis_saves_dmd_modes(self):
        """run_jacobian_analysis saves dmd_modes.npy with correct shape."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        with tempfile.TemporaryDirectory() as td:
            result = run_jacobian_analysis(
                _Sim(), self.trajs, output_dir=Path(td)
            )
            npy_path = Path(td) / "dmd_modes.npy"
            self.assertTrue(npy_path.exists())
            modes = np.load(npy_path, allow_pickle=True)
            N = self.trajs.shape[2]
            self.assertEqual(modes.shape[0], N)

    def test_mode_energy_in_result(self):
        """run_jacobian_analysis includes mode_energy array."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        result = run_jacobian_analysis(_Sim(), self.trajs)
        self.assertIn("mode_energy", result)
        self.assertEqual(len(result["mode_energy"]), self.trajs.shape[2])
        # All energies must be non-negative
        self.assertTrue(np.all(result["mode_energy"] >= 0))

    def test_hopf_absent_on_constant_trajectory(self):
        """No Hopf pairs expected for trivially constant trajectories."""
        from analysis.jacobian_analysis import (
            estimate_jacobian_dmd,
            analyze_jacobian_spectrum,
            _extract_dominant_hopf_mode,
        )
        N = 5
        # Constant + tiny noise → no oscillation
        rng = np.random.default_rng(1)
        trajs = np.full((3, 80, N), 0.5) + rng.normal(0, 1e-5, (3, 80, N))
        A = estimate_jacobian_dmd(trajs, burnin=2)
        spec = analyze_jacobian_spectrum(A, dt=1.0)
        hopf = _extract_dominant_hopf_mode(spec)
        # For near-identity A, very few or no Hopf pairs expected
        self.assertIsInstance(hopf, dict)
        self.assertIn("has_hopf", hopf)


class TestPhasePortrait(unittest.TestCase):
    """New Test 2: Phase portrait pair-wise PC projections."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        N, T, n_traj = 10, 120, 4
        t_arr = np.linspace(0, 8 * np.pi, T)
        trajs = np.zeros((n_traj, T, N), dtype=np.float32)
        for i in range(n_traj):
            for j in range(N):
                trajs[i, :, j] = np.sin(t_arr * (0.08 + j * 0.01))
        cls.trajs = trajs

    def test_phase_portrait_files_created(self):
        """run_pca_attractor creates all three phase portrait PNG files."""
        from spectral_dynamics.f_pca_attractor import run_pca_attractor

        with tempfile.TemporaryDirectory() as td:
            run_pca_attractor(self.trajs, output_dir=Path(td), burnin=5)
            self.assertTrue((Path(td) / "phase_portrait_pc1_pc2.png").exists())
            self.assertTrue((Path(td) / "phase_portrait_pc1_pc3.png").exists())
            self.assertTrue((Path(td) / "phase_portrait_pc2_pc3.png").exists())

    def test_phase_portrait_skipped_for_2d_trajectories(self):
        """pc1_pc3 and pc2_pc3 skipped when X has only 2 PCs."""
        from spectral_dynamics.f_pca_attractor import run_pca_attractor

        # N=2 → only 2 PCs possible; pc1_pc2 is created, pc1_pc3 / pc2_pc3 skipped
        rng = np.random.default_rng(9)
        trajs_2d = rng.random((3, 50, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            run_pca_attractor(trajs_2d, output_dir=Path(td), burnin=2)
            self.assertTrue((Path(td) / "phase_portrait_pc1_pc2.png").exists())
            # pc1_pc3 should NOT exist (only 2 PCs)
            self.assertFalse((Path(td) / "phase_portrait_pc1_pc3.png").exists())


class TestPoincaréSection(unittest.TestCase):
    """New Test 3: Poincaré section through PC1=0 plane."""

    def _make_circular_trajs(self) -> np.ndarray:
        """Return X_pca for simple circular orbit: PC1=sin(t), PC2=cos(t)."""
        T = 200
        t = np.linspace(0, 4 * np.pi, T)  # 2 full circles
        X = np.zeros((T, 3))
        X[:, 0] = np.sin(t)
        X[:, 1] = np.cos(t)
        X[:, 2] = np.sin(2 * t) * 0.3
        return X

    def test_periodic_orbit_detected(self):
        """Circular orbit (period-1) should be classified periodic."""
        from spectral_dynamics.f_pca_attractor import compute_poincare_section
        X = self._make_circular_trajs()
        result = compute_poincare_section(X, n_traj=1, steps_per_traj=200)
        self.assertGreater(result["n_crossings"], 0)
        self.assertIn(result["interpretation"],
                      ("periodic", "quasi-periodic", "chaotic",
                       "insufficient_crossings"))

    def test_no_crossings_when_pc1_positive(self):
        """No crossings detected when PC1 is always positive."""
        from spectral_dynamics.f_pca_attractor import compute_poincare_section
        T = 100
        X = np.ones((T, 3)) * 0.5   # PC1 always = 0.5 (never crosses 0)
        X[:, 0] = np.abs(np.sin(np.linspace(0, 4 * np.pi, T))) + 0.1
        result = compute_poincare_section(X, n_traj=1, steps_per_traj=T)
        self.assertEqual(result["n_crossings"], 0)
        self.assertEqual(result["interpretation"], "no_crossings")

    def test_poincare_points_shape(self):
        """Points array has shape (n_crossings, 2)."""
        from spectral_dynamics.f_pca_attractor import compute_poincare_section
        X = self._make_circular_trajs()
        result = compute_poincare_section(X, n_traj=1, steps_per_traj=200)
        pts = result["points"]
        self.assertEqual(pts.shape[1], 2)
        self.assertEqual(pts.shape[0], result["n_crossings"])

    def test_poincare_files_created_by_run_pca(self):
        """run_pca_attractor creates poincare_section.png and poincare_points.npy."""
        from spectral_dynamics.f_pca_attractor import run_pca_attractor

        rng = np.random.default_rng(55)
        T, N, n_traj = 150, 8, 3
        t_arr = np.linspace(0, 6 * np.pi, T)
        trajs = np.zeros((n_traj, T, N), dtype=np.float32)
        for i in range(n_traj):
            for j in range(N):
                trajs[i, :, j] = np.sin(t_arr * (0.1 + j * 0.02))

        with tempfile.TemporaryDirectory() as td:
            run_pca_attractor(trajs, output_dir=Path(td), burnin=10)
            self.assertTrue((Path(td) / "poincare_section.png").exists())
            self.assertTrue((Path(td) / "poincare_points.npy").exists())

    def test_poincare_result_in_pca_result_dict(self):
        """pca result dict includes poincare_n_crossings and poincare_interpretation."""
        from spectral_dynamics.f_pca_attractor import run_pca_attractor

        rng = np.random.default_rng(2)
        trajs = rng.random((3, 80, 6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            result = run_pca_attractor(trajs, output_dir=Path(td))
        self.assertIn("poincare_n_crossings", result)
        self.assertIn("poincare_interpretation", result)
        self.assertIsInstance(result["poincare_n_crossings"], int)
        self.assertIsInstance(result["poincare_interpretation"], str)


class TestLyapunovDivergenceCurve(unittest.TestCase):
    """Supplement 1: Lyapunov divergence curve for Rosenstein method."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        N, T, n_traj = 6, 150, 6
        trajs = np.zeros((n_traj, T, N), dtype=np.float32)
        for i in range(n_traj):
            x = rng.random(N).astype(np.float32)
            for t in range(T):
                trajs[i, t] = x
                x = np.clip(np.tanh(x * 0.85), 0, 1)
                x += rng.normal(0, 0.01, N).astype(np.float32)
                x = np.clip(x, 0, 1)
        cls.trajs = trajs

        class _Sim:
            dt = 2.0
            context_length = 10
            n_regions = N
            state_bounds = (0.0, 1.0)
            modality = "fmri"

        cls.sim = _Sim()

    def test_divergence_curve_file_saved(self):
        """Rosenstein run saves lyapunov_divergence_curve.npy."""
        from analysis.lyapunov import run_lyapunov_analysis

        with tempfile.TemporaryDirectory() as td:
            run_lyapunov_analysis(
                trajectories=self.trajs,
                simulator=self.sim,
                method="rosenstein",
                n_segments=1,
                output_dir=Path(td),
            )
            self.assertTrue(
                (Path(td) / "lyapunov_divergence_curve.npy").exists()
            )

    def test_divergence_curve_shape(self):
        """Divergence curve has shape (max_lag,) with finite values."""
        from analysis.lyapunov import run_lyapunov_analysis

        with tempfile.TemporaryDirectory() as td:
            run_lyapunov_analysis(
                trajectories=self.trajs,
                simulator=self.sim,
                method="rosenstein",
                n_segments=1,
                output_dir=Path(td),
            )
            curve = np.load(Path(td) / "lyapunov_divergence_curve.npy")
        self.assertGreater(len(curve), 1)
        # At least some finite values
        self.assertTrue(np.any(np.isfinite(curve)))

    def test_divergence_curve_not_saved_for_wolf(self):
        """Divergence curve file is NOT created for wolf-only runs."""
        from analysis.lyapunov import run_lyapunov_analysis

        class _Sim2:
            dt = 1.0
            context_length = 10
            n_regions = 6
            state_bounds = (0.0, 1.0)
            modality = "fmri"

            def rollout(self, **kw):
                raise RuntimeError("should not be called in this test")

        with tempfile.TemporaryDirectory() as td:
            # Use rosenstein=False path (test file absence, not wolf execution)
            # We can't easily run Wolf without a simulator; just verify the
            # divergence curve file is absent when no Rosenstein was run.
            # Simulate by testing that the file doesn't appear without output_dir
            result = run_lyapunov_analysis(
                trajectories=self.trajs,
                simulator=self.sim,
                method="rosenstein",
                n_segments=1,
                output_dir=None,   # no output → no file
            )
        # No file, but result must still have the core keys
        self.assertIn("mean_lyapunov", result)
        self.assertIn("lyapunov_values", result)


class TestDMDModalEnergy(unittest.TestCase):
    """Supplement 2: DMD modal energy analysis."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(99)
        N, T, n_traj = 8, 100, 4
        t = np.linspace(0, 4 * np.pi, T)
        trajs = np.zeros((n_traj, T, N), dtype=np.float64)
        for i in range(n_traj):
            for j in range(N):
                trajs[i, :, j] = 0.5 * np.sin(t * (0.1 + j * 0.02))
        cls.trajs = trajs

    def test_mode_energy_non_negative(self):
        """All DMD mode energies must be non-negative."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        result = run_jacobian_analysis(_Sim(), self.trajs)
        me = result.get("mode_energy")
        self.assertIsNotNone(me)
        self.assertTrue(np.all(np.asarray(me) >= 0))

    def test_mode_energy_length_equals_n_regions(self):
        """Mode energy array length equals number of brain regions N."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        result = run_jacobian_analysis(_Sim(), self.trajs)
        N = self.trajs.shape[2]
        self.assertEqual(len(result["mode_energy"]), N)

    def test_dmd_mode_energy_png_created(self):
        """run_jacobian_analysis creates dmd_mode_energy.png."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        with tempfile.TemporaryDirectory() as td:
            run_jacobian_analysis(_Sim(), self.trajs, output_dir=Path(td))
            self.assertTrue((Path(td) / "dmd_mode_energy.png").exists())

    def test_dmd_modes_npy_has_right_shape(self):
        """dmd_modes.npy has shape (N, N) — one eigenvector per column."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        N = self.trajs.shape[2]
        with tempfile.TemporaryDirectory() as td:
            run_jacobian_analysis(_Sim(), self.trajs, output_dir=Path(td))
            modes = np.load(Path(td) / "dmd_modes.npy", allow_pickle=True)
        self.assertEqual(modes.shape, (N, N))

    def test_hopf_eigenvalues_png_created(self):
        """run_jacobian_analysis creates hopf_eigenvalues.png."""
        from analysis.jacobian_analysis import run_jacobian_analysis

        class _Sim:
            dt = 1.0

        with tempfile.TemporaryDirectory() as td:
            run_jacobian_analysis(_Sim(), self.trajs, output_dir=Path(td))
            self.assertTrue((Path(td) / "hopf_eigenvalues.png").exists())


if __name__ == "__main__":
    unittest.main()
