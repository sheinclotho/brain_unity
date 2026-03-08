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

_TD_DIR = _REPO_ROOT / "twinbrain-dynamics"
_SD_DIR = _REPO_ROOT / "spectral_dynamics"
for _p in (_TD_DIR, _SD_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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


if __name__ == "__main__":
    unittest.main()
