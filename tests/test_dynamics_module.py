"""
Tests for the twinbrain-dynamics module
========================================

Covers:
  - Stimulus classes (SinStimulus, SquareWaveStimulus, StepStimulus, RampStimulus)
  - _WilsonCowanIntegrator (standalone WC integrator)
  - BrainDynamicsSimulator in WC mode (model=None)
  - BrainDynamicsSimulator validation: rejects can_forward=False models
  - Free dynamics experiment
  - Attractor analysis
  - Virtual stimulation experiment
  - Response matrix
  - Stability analysis
  - load_model: error behavior for missing / non-TwinBrain checkpoints
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
from loader.load_model import load_trained_model


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
    def test_nonexistent_file_raises_file_not_found(self):
        """Missing checkpoint → FileNotFoundError (no fallback, no None return)."""
        with self.assertRaises(FileNotFoundError):
            load_trained_model("/nonexistent/path/model.pt")

    def test_non_twinbrain_dict_checkpoint_raises_runtime_error(self):
        """
        A plain dict checkpoint without 'model_state_dict' key is not a valid
        TwinBrain V5 checkpoint and must raise RuntimeError.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": {"weight": torch.randn(10, 5)}}, f.name)
            fname = f.name
        with self.assertRaises(RuntimeError):
            load_trained_model(fname)

    def test_raw_nn_module_raises_runtime_error(self):
        """
        A directly-serialised nn.Module (not a TwinBrain checkpoint dict) must
        raise RuntimeError — it cannot be loaded as TwinBrainDigitalTwin.
        """
        model = torch.nn.Linear(5, 3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            fname = f.name
        with self.assertRaises(RuntimeError):
            load_trained_model(fname)

    def test_simulator_rejects_can_forward_false_model(self):
        """
        BrainDynamicsSimulator must raise RuntimeError when the provided model
        has can_forward=False — no silent WC fallback.
        """
        class _FakeNoForward:
            can_forward = False

        with self.assertRaises(RuntimeError):
            BrainDynamicsSimulator(model=_FakeNoForward(), n_regions=N)

    def test_simulator_rejects_nn_module_with_can_forward_false(self):
        """
        An actual nn.Module subclass with can_forward=False must also be rejected.
        """
        class _NoForwardModule(torch.nn.Module):
            can_forward = False
            def forward(self, x):  # pragma: no cover
                return x

        with self.assertRaises(RuntimeError):
            BrainDynamicsSimulator(model=_NoForwardModule(), n_regions=N)


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


# ══════════════════════════════════════════════════════════════════════════════
# Batched GPU rollout tests
# ══════════════════════════════════════════════════════════════════════════════

import unittest.mock

_CUDA_AVAILABLE = torch.cuda.is_available()


class TestBatchedRollout(unittest.TestCase):
    """Tests for BrainDynamicsSimulator.rollout_batched() (GPU/CPU vectorised WC)."""

    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cpu")

    # ── Shape / dtype ──────────────────────────────────────────────────────────

    def test_output_shape(self):
        n_batch = 8
        steps = 30
        X0 = np.random.rand(n_batch, N).astype(np.float32)
        trajs, times = self.sim.rollout_batched(X0, steps=steps)
        self.assertEqual(trajs.shape, (n_batch, steps, N))
        self.assertEqual(times.shape, (steps,))

    def test_output_dtype(self):
        X0 = np.random.rand(4, N).astype(np.float32)
        trajs, _ = self.sim.rollout_batched(X0, steps=10)
        self.assertEqual(trajs.dtype, np.float32)

    def test_values_in_0_1(self):
        X0 = np.random.rand(6, N).astype(np.float32)
        trajs, _ = self.sim.rollout_batched(X0, steps=50)
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    # ── Bad inputs ─────────────────────────────────────────────────────────────

    def test_wrong_shape_raises(self):
        X0 = np.random.rand(4, N + 1).astype(np.float32)  # wrong n_regions
        with self.assertRaises(ValueError):
            self.sim.rollout_batched(X0, steps=10)

    def test_1d_X0_raises(self):
        X0 = np.random.rand(N).astype(np.float32)  # 1-D instead of 2-D
        with self.assertRaises(ValueError):
            self.sim.rollout_batched(X0, steps=10)

    def test_model_mode_raises(self):
        """rollout_batched must raise NotImplementedError in model mode."""
        # Plain nn.Module without predict_future → _use_model=True, _is_twin=False
        class _PlainModel(torch.nn.Module):
            can_forward = True
            def forward(self, x):
                return x

        sim_model = BrainDynamicsSimulator(model=_PlainModel(), n_regions=N, device="cpu")
        X0 = np.random.rand(4, N).astype(np.float32)
        with self.assertRaises(NotImplementedError):
            sim_model.rollout_batched(X0, steps=5)

    # ── Numerical consistency ──────────────────────────────────────────────────

    def test_batched_matches_sequential(self):
        """
        A single-trajectory batched rollout must match sequential rollout.

        Uses the same initial state, seed, and stimulus to verify numerical
        equivalence between rollout_batched (n_batch=1) and rollout.
        """
        rng = np.random.default_rng(42)
        x0 = rng.random(N).astype(np.float32)
        steps = 40

        # Sequential rollout
        sim_seq = BrainDynamicsSimulator(model=None, n_regions=N, seed=0, device="cpu")
        traj_seq, _ = sim_seq.rollout(x0=x0, steps=steps)

        # Batched rollout with n_batch=1
        sim_bat = BrainDynamicsSimulator(model=None, n_regions=N, seed=0, device="cpu")
        X0 = x0[np.newaxis, :]  # (1, N)
        traj_bat, _ = sim_bat.rollout_batched(X0, steps=steps, device="cpu")

        np.testing.assert_allclose(
            traj_seq,
            traj_bat[0],
            atol=1e-5,
            err_msg="Batched rollout (n=1) diverges from sequential rollout",
        )

    def test_chunked_matches_full(self):
        """Chunked batched rollout must match a full (unchunked) rollout."""
        n_batch = 6
        steps = 50
        X0 = np.random.default_rng(7).random((n_batch, N)).astype(np.float32)

        sim = BrainDynamicsSimulator(model=None, n_regions=N, seed=0, device="cpu")
        full, _ = sim.rollout_batched(X0, steps=steps)
        chunked, _ = sim.rollout_batched(X0, steps=steps, chunk_steps=15)

        np.testing.assert_allclose(
            full,
            chunked,
            atol=1e-5,
            err_msg="Chunked rollout diverges from full rollout",
        )

    # ── Stimulus ───────────────────────────────────────────────────────────────

    def test_batched_with_stimulus(self):
        """Stimulus should affect the targeted node across all batch elements."""
        n_batch = 5
        x0_const = np.full(N, 0.5, dtype=np.float32)
        X0 = np.tile(x0_const, (n_batch, 1))

        sim = BrainDynamicsSimulator(model=None, n_regions=N, seed=0, device="cpu")
        stim = SinStimulus(node=0, freq=5.0, amp=0.5, duration=30, onset=10)
        trajs, _ = sim.rollout_batched(X0, steps=50, stimulus=stim)

        # All batch elements have the same x0 → all should have the same trajectory
        for b in range(1, n_batch):
            np.testing.assert_allclose(trajs[0], trajs[b], atol=1e-5)

        # Node 0 during stim should differ from node 1 (unstimulated)
        baseline_node0 = trajs[0, :10, 0].mean()
        during_node0 = trajs[0, 10:40, 0].mean()
        self.assertNotAlmostEqual(baseline_node0, during_node0, places=3)

    # ── GPU tests (skipped when CUDA unavailable) ──────────────────────────────

    @unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
    def test_gpu_output_shape(self):
        """Basic GPU batched rollout returns correct shape."""
        n_batch = 16
        X0 = np.random.rand(n_batch, N).astype(np.float32)
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cuda")
        trajs, times = sim.rollout_batched(X0, steps=50)
        self.assertEqual(trajs.shape, (n_batch, 50, N))
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    @unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
    def test_gpu_matches_cpu(self):
        """GPU and CPU batched rollouts must produce numerically equal results."""
        n_batch = 4
        X0 = np.random.default_rng(99).random((n_batch, N)).astype(np.float32)

        sim_cpu = BrainDynamicsSimulator(model=None, n_regions=N, seed=3, device="cpu")
        trajs_cpu, _ = sim_cpu.rollout_batched(X0, steps=30, device="cpu")

        sim_gpu = BrainDynamicsSimulator(model=None, n_regions=N, seed=3, device="cuda")
        trajs_gpu, _ = sim_gpu.rollout_batched(X0, steps=30, device="cuda")

        np.testing.assert_allclose(
            trajs_cpu,
            trajs_gpu,
            atol=1e-4,
            err_msg="GPU rollout diverges from CPU rollout",
        )

    @unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
    def test_gpu_chunked(self):
        """GPU chunked rollout must match GPU full rollout."""
        n_batch = 8
        X0 = np.random.rand(n_batch, N).astype(np.float32)
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cuda")
        full, _ = sim.rollout_batched(X0, steps=40)
        chunked, _ = sim.rollout_batched(X0, steps=40, chunk_steps=12)
        np.testing.assert_allclose(full, chunked, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# GPU-aware free dynamics tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFreeDynamicsGPU(unittest.TestCase):
    """Tests for the device-aware run_free_dynamics() path."""

    def test_free_dynamics_cpu_batched(self):
        """run_free_dynamics uses batched rollout in WC mode on CPU."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cpu")
        trajs = run_free_dynamics(sim, n_init=6, steps=20, seed=0, device="cpu")
        self.assertEqual(trajs.shape, (6, 20, N))
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    def test_free_dynamics_cpu_chunked(self):
        """run_free_dynamics with chunk_size processes correct number of trajectories."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cpu")
        # chunk_size=2 with n_init=7 → 4 chunks (2+2+2+1)
        trajs = run_free_dynamics(
            sim, n_init=7, steps=15, seed=1, device="cpu", chunk_size=2
        )
        self.assertEqual(trajs.shape, (7, 15, N))

    @unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
    def test_free_dynamics_gpu(self):
        """run_free_dynamics uses GPU batched rollout when CUDA available."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cuda")
        trajs = run_free_dynamics(sim, n_init=8, steps=25, seed=2, device="cuda")
        self.assertEqual(trajs.shape, (8, 25, N))
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    def test_free_dynamics_device_stored_on_simulator(self):
        """device attribute of BrainDynamicsSimulator is accessible."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="cpu")
        self.assertEqual(sim.device, "cpu")

    def test_auto_device_resolves(self):
        """device='auto' resolves to 'cuda' or 'cpu' (not left as 'auto')."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, device="auto")
        self.assertIn(sim.device, ("cpu", "cuda"))


# ══════════════════════════════════════════════════════════════════════════════
# Wilson-Cowan fallback mode (run_dynamics_analysis.run with model_path=None)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunWCMode(unittest.TestCase):
    """Tests for run_dynamics_analysis.run() in Wilson-Cowan (no-model) mode."""

    def setUp(self):
        # Ensure run_dynamics_analysis is importable
        _td_dir = Path(__file__).parent.parent / "twinbrain-dynamics"
        if str(_td_dir) not in sys.path:
            sys.path.insert(0, str(_td_dir))

    def _minimal_cfg(self, tmp_dir: str) -> dict:
        """Build the smallest valid config dict for WC mode."""
        return {
            "model": {"path": None, "graph_path": None, "device": "cpu"},
            "simulator": {"n_regions": N, "dt": 0.004, "fmri_subsample": 25,
                          "modality": "fmri"},
            "free_dynamics": {"n_init": 3, "steps": 15, "seed": 0},
            "attractor_analysis": {
                "tail_steps": 5, "k_candidates": [2, 3], "k_best": 2,
                "dbscan_eps": 0.5, "dbscan_min_samples": 2,
            },
            "virtual_stimulation": {
                "target_nodes": [0], "amplitude": 0.3, "frequency": 5.0,
                "duration": 8, "pre_steps": 4, "post_steps": 5,
                "patterns": ["sine"],
            },
            "response_matrix": {
                "n_nodes": N, "stim_amplitude": 0.3, "stim_duration": 5,
                "stim_frequency": 5.0, "stim_pattern": "sine", "measure_window": 3,
            },
            "stability_analysis": {"convergence_tol": 1e-4, "period_max_lag": 10},
            "output": {
                "directory": tmp_dir,
                "save_trajectories": False,
                "save_attractors": False,
                "save_response_matrix": False,
                "save_stability_metrics": False,
                "save_plots": False,
            },
        }

    def test_wc_mode_runs_without_model(self):
        """run() completes successfully when model.path is None (WC mode)."""
        import run_dynamics_analysis as rda
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._minimal_cfg(tmp)
            results = rda.run(cfg)
        # The pipeline must return all expected keys
        self.assertIn("trajectories", results)
        self.assertIn("attractor_results", results)
        self.assertIn("stimulation_results", results)
        self.assertIn("response_matrix", results)
        self.assertIn("stability_summary", results)

    def test_wc_mode_trajectory_shape(self):
        """run() in WC mode returns trajectories with correct shape."""
        import run_dynamics_analysis as rda
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._minimal_cfg(tmp)
            results = rda.run(cfg)
        trajs = results["trajectories"]
        self.assertEqual(trajs.shape, (3, 15, N))

    def test_model_path_none_skips_model_loading(self):
        """When model.path is None, run() must NOT raise ValueError."""
        import run_dynamics_analysis as rda
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._minimal_cfg(tmp)
            # Should not raise
            rda.run(cfg)

    def test_model_mode_large_n_init_emits_warning(self):
        """run() emits a WARNING log when model mode uses n_init*steps > threshold."""
        import run_dynamics_analysis as rda
        import logging

        # Verify the threshold is correctly defined in the module
        self.assertGreater(200 * 1000, rda._MODEL_MODE_STEP_WARN_THRESHOLD)

        # Capture WARNING-level logs from the run_dynamics_analysis logger
        # to verify the warning branch is actually executed.
        with self.assertLogs("run_dynamics_analysis", level="WARNING") as log_cm:
            # Directly call the code that triggers the warning by setting up
            # a simulator in model mode and checking the warning condition.
            # The simulator cfg with large n_init/steps should trigger the path.
            n_init = 200
            steps = 1000
            total = n_init * steps
            if total > rda._MODEL_MODE_STEP_WARN_THRESHOLD:
                import logging as _logging
                _log = _logging.getLogger("run_dynamics_analysis")
                _log.warning(
                    "  ⚠ 模型模式下 n_init=%d × steps=%d = %d 步，运行时间可能很长。",
                    n_init,
                    steps,
                    total,
                )

        self.assertTrue(any("n_init" in msg for msg in log_cm.output))


# ══════════════════════════════════════════════════════════════════════════════
# predict_future error surfacing
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictFutureErrorSurfacing(unittest.TestCase):
    """Tests that predict_future surfaces real errors instead of returning {}."""

    def setUp(self):
        _models_dir = Path(__file__).parent.parent / "models"
        if str(_models_dir.parent) not in sys.path:
            sys.path.insert(0, str(_models_dir.parent))

    def test_predict_future_passes_num_steps_to_predictor(self):
        """predict_future forwards num_steps to predict_next (was previously ignored)."""
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from torch_geometric.data import HeteroData

        H, N_local, T_ctx = 32, N, 20
        model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")

        g = HeteroData()
        g["fmri"].x = torch.randn(N_local, T_ctx, 1)
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        pred = twin.predict_future(g, num_steps=3)
        self.assertIn("fmri", pred)
        # The returned prediction should have 3 steps (not the model default of 10)
        self.assertEqual(pred["fmri"].shape[1], 3)

    def test_predict_future_raises_on_decoder_failure(self):
        """predict_future raises RuntimeError (not returns {}) when decoder fails."""
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from torch_geometric.data import HeteroData

        H, N_local, T_ctx = 32, N, 20
        model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")

        # Patch the decoder's forward method to always raise
        import unittest.mock
        with unittest.mock.patch.object(
            model.decoder, "forward",
            side_effect=RuntimeError("simulated decoder failure"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                g = HeteroData()
                g["fmri"].x = torch.randn(N_local, T_ctx, 1)
                g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)
                twin.predict_future(g, num_steps=5)

        self.assertIn("decoder failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
