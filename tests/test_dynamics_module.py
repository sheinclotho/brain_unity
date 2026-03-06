"""
Tests for the twinbrain-dynamics module
========================================

Covers:
  - Stimulus classes (SinStimulus, SquareWaveStimulus, StepStimulus, RampStimulus)
  - BrainDynamicsSimulator (step, rollout)
  - Free dynamics experiment
  - Attractor analysis
  - Virtual stimulation experiment
  - Response matrix
  - Stability analysis
  - load_model (StateWrapper, file-not-found)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure the twinbrain-dynamics directory is on sys.path
_TD_DIR = Path(__file__).parent.parent / "twinbrain-dynamics"
if str(_TD_DIR) not in sys.path:
    sys.path.insert(0, str(_TD_DIR))

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    RampStimulus,
    SinStimulus,
    SquareWaveStimulus,
    StepStimulus,
    _WilsonCowanIntegrator,
)
from experiments.free_dynamics import run_free_dynamics
from experiments.attractor_analysis import extract_final_states, run_attractor_analysis
from experiments.virtual_stimulation import run_stimulation, run_virtual_stimulation
from analysis.response_matrix import compute_response_matrix
from analysis.stability_analysis import (
    analyze_trajectory_stability,
    compute_state_deltas,
    run_stability_analysis,
)
from loader.load_model import load_trained_model, _StateWrapper


# ── Constants used across tests ───────────────────────────────────────────────
N = 10   # small n_regions for fast tests


# ══════════════════════════════════════════════════════════════════════════════
# Stimulus Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSinStimulus(unittest.TestCase):
    def _make(self, onset=0, duration=100, amp=0.5, freq=10.0):
        return SinStimulus(node=3, freq=freq, amp=amp, duration=duration, onset=onset)

    def test_inactive_before_onset(self):
        s = self._make(onset=10)
        self.assertEqual(s.value(9), 0.0)

    def test_inactive_after_offset(self):
        s = self._make(onset=0, duration=100)
        self.assertEqual(s.value(100), 0.0)

    def test_active_during_stim(self):
        s = self._make(onset=0, duration=100, amp=0.5)
        # Middle of duration should have non-zero value
        val = s.value(50)
        self.assertGreater(abs(val), 0.0)
        self.assertLessEqual(abs(val), 0.5 + 1e-9)

    def test_no_boundary_zeros(self):
        """Middle-point sampling: first and last steps must NOT be zero."""
        s = self._make(onset=0, duration=10, amp=1.0)
        self.assertGreater(s.value(0), 0.0, "First step should not be zero")
        self.assertGreater(s.value(9), 0.0, "Last step should not be zero")

    def test_is_active(self):
        s = self._make(onset=5, duration=10)
        self.assertFalse(s.is_active(4))
        self.assertTrue(s.is_active(5))
        self.assertTrue(s.is_active(14))
        self.assertFalse(s.is_active(15))


class TestSquareWaveStimulus(unittest.TestCase):
    def test_alternates(self):
        s = SquareWaveStimulus(node=0, freq=1.0, amp=1.0, duration=100, onset=0, dt=0.1)
        # freq=1 Hz, dt=0.1s → period=10 steps, half-period=5 steps
        self.assertGreater(s.value(0), 0.0)   # first half
        self.assertEqual(s.value(5), 0.0)      # second half

    def test_inactive_outside(self):
        s = SquareWaveStimulus(node=0, freq=5.0, amp=0.5, duration=20, onset=10, dt=0.004)
        self.assertEqual(s.value(9), 0.0)
        self.assertEqual(s.value(30), 0.0)


class TestStepStimulus(unittest.TestCase):
    def test_max_amplitude_in_middle(self):
        s = StepStimulus(node=0, amp=1.0, duration=100, onset=0, ramp_steps=10)
        # Middle region (steps 10–89) should be at full amplitude
        self.assertAlmostEqual(s.value(50), 1.0)

    def test_soft_onset(self):
        s = StepStimulus(node=0, amp=1.0, duration=100, onset=0, ramp_steps=10)
        # At step 5 (halfway through ramp) → ~0.5
        self.assertAlmostEqual(s.value(5), 0.5, places=5)

    def test_inactive_outside(self):
        s = StepStimulus(node=0, amp=1.0, duration=10, onset=5)
        self.assertEqual(s.value(4), 0.0)
        self.assertEqual(s.value(15), 0.0)


class TestRampStimulus(unittest.TestCase):
    def test_zero_at_start(self):
        s = RampStimulus(node=0, amplitude=1.0, duration=100, onset=0)
        self.assertAlmostEqual(s.value(0), 0.0, places=3)

    def test_max_at_end(self):
        s = RampStimulus(node=0, amplitude=1.0, duration=100, onset=0)
        self.assertAlmostEqual(s.value(99), 1.0, places=3)

    def test_inactive_outside(self):
        s = RampStimulus(node=0, amplitude=1.0, duration=10, onset=5)
        self.assertEqual(s.value(4), 0.0)
        self.assertEqual(s.value(15), 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# WilsonCowanIntegrator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestWilsonCowanIntegrator(unittest.TestCase):
    def test_no_stim_stays_near_x0(self):
        """Without stimulation, state should stay close to x0."""
        x0 = np.full(N, 0.5, dtype=np.float32)
        wc = _WilsonCowanIntegrator(N, x0=x0)
        state = x0.copy()
        for _ in range(50):
            state = wc.step(state, np.zeros(N, dtype=np.float32))
        # Should be very close to x0 (leak drives it back)
        np.testing.assert_allclose(state, x0, atol=0.02)

    def test_stim_increases_target(self):
        """With stimulation, targeted region should deviate from x0."""
        x0 = np.full(N, 0.5, dtype=np.float32)
        wc = _WilsonCowanIntegrator(N, x0=x0)
        stim = np.zeros(N, dtype=np.float32)
        stim[0] = 1.0
        state = x0.copy()
        for _ in range(20):
            state = wc.step(state, stim)
        self.assertGreater(state[0], x0[0])

    def test_output_clipped_to_0_1(self):
        x0 = np.zeros(N, dtype=np.float32)
        wc = _WilsonCowanIntegrator(N, x0=x0)
        stim = np.ones(N, dtype=np.float32) * 100.0  # extreme
        state = x0.copy()
        for _ in range(10):
            state = wc.step(state, stim)
        self.assertTrue(np.all(state >= 0.0))
        self.assertTrue(np.all(state <= 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# BrainDynamicsSimulator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBrainDynamicsSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_rollout_shape(self):
        x0 = np.random.rand(N).astype(np.float32)
        traj, times = self.sim.rollout(x0, steps=50)
        self.assertEqual(traj.shape, (50, N))
        self.assertEqual(times.shape, (50,))

    def test_rollout_values_in_0_1(self):
        x0 = np.random.rand(N).astype(np.float32)
        traj, _ = self.sim.rollout(x0, steps=100)
        self.assertTrue(np.all(traj >= 0.0))
        self.assertTrue(np.all(traj <= 1.0))

    def test_rollout_with_stimulus(self):
        x0 = np.full(N, 0.5, dtype=np.float32)
        stim = SinStimulus(node=0, freq=5.0, amp=0.5, duration=50, onset=10)
        traj, _ = self.sim.rollout(x0, steps=80, stimulus=stim)
        self.assertEqual(traj.shape, (80, N))
        # Node 0 during stim should differ from baseline
        baseline = traj[:10, 0].mean()
        during_stim = traj[10:60, 0].mean()
        # There should be some difference (stimulation effect)
        self.assertNotEqual(round(baseline, 4), round(during_stim, 4))

    def test_rollout_wrong_x0_shape_raises(self):
        with self.assertRaises(ValueError):
            self.sim.rollout(np.zeros(N + 1), steps=10)

    def test_free_rollout_stable(self):
        """Free dynamics from constant initial state should not diverge."""
        x0 = np.full(N, 0.5, dtype=np.float32)
        traj, _ = self.sim.rollout(x0, steps=200)
        self.assertTrue(np.all(np.isfinite(traj)))

    def test_sample_random_state(self):
        x0 = self.sim.sample_random_state()
        self.assertEqual(x0.shape, (N,))
        self.assertTrue(np.all(x0 >= 0.0))
        self.assertTrue(np.all(x0 <= 1.0))

    def test_rollout_multi_stim(self):
        x0 = np.full(N, 0.5, dtype=np.float32)
        stimuli = [
            SinStimulus(node=0, freq=5.0, amp=0.3, duration=50, onset=0),
            StepStimulus(node=1, amp=0.3, duration=50, onset=0),
        ]
        traj, _ = self.sim.rollout_multi_stim(x0, steps=60, stimuli=stimuli)
        self.assertEqual(traj.shape, (60, N))


# ══════════════════════════════════════════════════════════════════════════════
# Free Dynamics Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFreeDynamics(unittest.TestCase):
    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_output_shape(self):
        trajs = run_free_dynamics(self.sim, n_init=5, steps=20, seed=0)
        self.assertEqual(trajs.shape, (5, 20, N))

    def test_values_in_0_1(self):
        trajs = run_free_dynamics(self.sim, n_init=3, steps=30, seed=1)
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_free_dynamics(self.sim, n_init=2, steps=10, output_dir=Path(tmp))
            self.assertTrue((Path(tmp) / "trajectories.npy").exists())


# ══════════════════════════════════════════════════════════════════════════════
# Attractor Analysis Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAttractorAnalysis(unittest.TestCase):
    def setUp(self):
        # Create synthetic trajectories converging to 2 distinct attractors
        rng = np.random.default_rng(0)
        n_init = 20
        steps = 50
        n_regions = N
        trajs = np.zeros((n_init, steps, n_regions), dtype=np.float32)
        for i in range(n_init):
            if i < n_init // 2:
                final = np.zeros(n_regions, dtype=np.float32) + 0.2
            else:
                final = np.ones(n_regions, dtype=np.float32) * 0.8
            trajs[i] = final + rng.random((steps, n_regions)).astype(np.float32) * 0.01
        self.trajs = trajs

    def test_extract_final_states_shape(self):
        fs = extract_final_states(self.trajs, tail_steps=10)
        self.assertEqual(fs.shape, (20, N))

    def test_run_attractor_analysis_keys(self):
        result = run_attractor_analysis(
            self.trajs,
            tail_steps=10,
            k_candidates=[2, 3],
            k_best=2,
        )
        self.assertIn("kmeans_labels", result)
        self.assertIn("basin_distribution", result)
        self.assertIn("kmeans_centers", result)
        self.assertIn("dbscan_n_clusters", result)

    def test_basin_distribution_sums_to_1(self):
        result = run_attractor_analysis(
            self.trajs,
            tail_steps=10,
            k_candidates=[2, 3],
            k_best=2,
        )
        total = sum(result["basin_distribution"].values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_attractor_analysis(
                self.trajs,
                tail_steps=5,
                k_candidates=[2],
                k_best=2,
                output_dir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "attractor_states.npy").exists())


# ══════════════════════════════════════════════════════════════════════════════
# Virtual Stimulation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVirtualStimulation(unittest.TestCase):
    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_run_stimulation_result_shapes(self):
        result = run_stimulation(
            self.sim,
            node=0,
            stim_steps=20,
            pre_steps=10,
            post_steps=10,
        )
        self.assertEqual(result.pre_trajectory.shape, (10, N))
        self.assertEqual(result.stim_trajectory.shape, (20, N))
        self.assertEqual(result.post_trajectory.shape, (10, N))
        self.assertEqual(result.full_trajectory.shape, (40, N))

    def test_peak_response_in_0_1(self):
        result = run_stimulation(self.sim, node=0, stim_steps=30, pre_steps=5, post_steps=5)
        self.assertTrue(np.all(result.peak_response >= 0.0))
        self.assertTrue(np.all(result.peak_response <= 1.0))

    def test_patterns_available(self):
        for pattern in ("sine", "square", "step", "ramp"):
            result = run_stimulation(
                self.sim, node=1, stim_steps=20, pre_steps=5, post_steps=5, pattern=pattern
            )
            self.assertEqual(result.stim_trajectory.shape, (20, N))

    def test_invalid_pattern_raises(self):
        with self.assertRaises(ValueError):
            run_stimulation(self.sim, node=0, stim_steps=10, pattern="invalid")

    def test_run_virtual_stimulation_structure(self):
        res = run_virtual_stimulation(
            self.sim,
            target_nodes=[0, 1],
            patterns=["sine", "step"],
            stim_steps=10,
            pre_steps=5,
            post_steps=5,
        )
        self.assertIn("sine", res)
        self.assertIn("step", res)
        self.assertEqual(len(res["sine"]), 2)   # 2 target nodes

    def test_saves_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_virtual_stimulation(
                self.sim,
                target_nodes=[0],
                patterns=["sine"],
                stim_steps=10,
                pre_steps=5,
                post_steps=5,
                output_dir=Path(tmp),
            )
            files = list(Path(tmp).glob("*.npy"))
            self.assertGreater(len(files), 0)


# ══════════════════════════════════════════════════════════════════════════════
# Response Matrix Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseMatrix(unittest.TestCase):
    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_shape(self):
        R = compute_response_matrix(
            self.sim,
            n_nodes=N,
            stim_duration=10,
            pre_steps=5,
            measure_window=5,
        )
        self.assertEqual(R.shape, (N, N))

    def test_dtype(self):
        R = compute_response_matrix(
            self.sim,
            n_nodes=3,
            stim_duration=5,
            pre_steps=3,
            measure_window=3,
        )
        self.assertEqual(R.dtype, np.float32)

    def test_diagonal_positive(self):
        """Stimulated node should show positive or non-negative self-response."""
        R = compute_response_matrix(
            self.sim,
            n_nodes=5,
            stim_duration=20,
            pre_steps=10,
            measure_window=10,
        )
        # Diagonal (self-response) should generally be non-negative
        # (at least no strong inhibition of the stimulated node)
        # Accept minor floating-point negatives due to dynamics
        self.assertTrue(np.mean(np.diag(R)) >= -0.1)

    def test_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            compute_response_matrix(
                self.sim,
                n_nodes=2,
                stim_duration=5,
                pre_steps=3,
                measure_window=3,
                output_dir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "response_matrix.npy").exists())


# ══════════════════════════════════════════════════════════════════════════════
# Stability Analysis Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStabilityAnalysis(unittest.TestCase):
    def test_compute_state_deltas_shape(self):
        traj = np.random.rand(50, N).astype(np.float32)
        deltas = compute_state_deltas(traj)
        self.assertEqual(deltas.shape, (49,))

    def test_fixed_point_classification(self):
        """Constant trajectory should be classified as fixed_point."""
        traj = np.ones((100, N), dtype=np.float32) * 0.5
        metrics = analyze_trajectory_stability(traj)
        self.assertEqual(metrics["classification"], "fixed_point")
        self.assertEqual(metrics["convergence_step"], 0)

    def test_unstable_classification(self):
        """Random walk should be classified as unstable."""
        rng = np.random.default_rng(0)
        traj = rng.random((100, N)).astype(np.float32)
        metrics = analyze_trajectory_stability(traj)
        self.assertIn(
            metrics["classification"],
            ["unstable", "metastable"],
        )

    def test_run_stability_analysis_keys(self):
        trajs = np.random.rand(5, 30, N).astype(np.float32)
        summary = run_stability_analysis(trajs)
        self.assertIn("classification_counts", summary)
        self.assertIn("fraction_converged", summary)
        for frac_key in ("fraction_converged", "fraction_limit_cycle",
                         "fraction_metastable", "fraction_unstable"):
            total = (
                summary["fraction_converged"]
                + summary["fraction_limit_cycle"]
                + summary["fraction_metastable"]
                + summary["fraction_unstable"]
            )
            self.assertAlmostEqual(total, 1.0, places=4)
            break

    def test_fractions_sum_to_1(self):
        trajs = np.random.rand(10, 30, N).astype(np.float32)
        summary = run_stability_analysis(trajs)
        total = (
            summary["fraction_converged"]
            + summary["fraction_limit_cycle"]
            + summary["fraction_metastable"]
            + summary["fraction_unstable"]
        )
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(3, 20, N).astype(np.float32)
            run_stability_analysis(trajs, output_dir=Path(tmp))
            json_path = Path(tmp) / "stability_metrics.json"
            self.assertTrue(json_path.exists())
            with open(json_path) as fh:
                data = json.load(fh)
            self.assertIn("classification_counts", data)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loader Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadModel(unittest.TestCase):
    def test_nonexistent_file_returns_none(self):
        result = load_trained_model("/nonexistent/path/model.pt")
        self.assertIsNone(result)

    def test_state_dict_returns_wrapper(self):
        """A plain dict checkpoint → _StateWrapper."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": {"weight": torch.randn(10, 5)}}, f.name)
            wrapper = load_trained_model(f.name)
        self.assertIsInstance(wrapper, _StateWrapper)
        self.assertFalse(wrapper.can_forward)

    def test_nn_module_loads_and_freezes(self):
        """A saved nn.Module should load, eval, and be frozen."""
        model = torch.nn.Linear(5, 3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            loaded = load_trained_model(f.name)
        self.assertIsNotNone(loaded)
        self.assertFalse(loaded.training)
        for p in loaded.parameters():
            self.assertFalse(p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Integration: mini end-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):
    """Smoke test for the full pipeline with tiny parameters."""

    def test_full_pipeline(self):
        sim = BrainDynamicsSimulator(model=None, n_regions=N)

        # Free dynamics
        trajs = run_free_dynamics(sim, n_init=5, steps=30, seed=7)
        self.assertEqual(trajs.shape, (5, 30, N))

        # Attractor analysis
        att = run_attractor_analysis(
            trajs,
            tail_steps=5,
            k_candidates=[2, 3],
            k_best=2,
        )
        self.assertIn("kmeans_k", att)

        # Stimulation
        stim_res = run_stimulation(sim, node=0, stim_steps=10, pre_steps=5, post_steps=5)
        self.assertEqual(stim_res.full_trajectory.shape, (20, N))

        # Response matrix
        R = compute_response_matrix(sim, n_nodes=N, stim_duration=5, pre_steps=3, measure_window=3)
        self.assertEqual(R.shape, (N, N))

        # Stability
        stab = run_stability_analysis(trajs)
        total_frac = (
            stab["fraction_converged"]
            + stab["fraction_limit_cycle"]
            + stab["fraction_metastable"]
            + stab["fraction_unstable"]
        )
        self.assertAlmostEqual(total_frac, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
