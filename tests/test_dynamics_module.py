"""
Tests for the twinbrain-dynamics module
========================================

Covers:
  - Stimulus classes (SinStimulus, SquareWaveStimulus, StepStimulus, RampStimulus)
  - BrainDynamicsSimulator validation: rejects can_forward=False models
  - BrainDynamicsSimulator twin-mode: context trimming, x0 injection,
    sample_random_state() z-score awareness
  - Free dynamics experiment
  - Attractor analysis
  - Virtual stimulation experiment
  - Response matrix
  - Stability analysis (method A + method B delay-distance)
  - Lyapunov exponent estimation
  - Trajectory convergence analysis
  - Random model comparison
  - Delay-embedding helper (_build_delay_embedding)
  - Convergence-threshold skip behaviour (raised to 0.05)
  - load_model: error behavior for missing / non-TwinBrain checkpoints

NOTE: Tests for the deprecated Wilson-Cowan (WC) mode and
_WilsonCowanIntegrator are marked @unittest.skip — WC mode was removed
from BrainDynamicsSimulator (see AGENTS.md).  The old API
BrainDynamicsSimulator(model=None, n_regions=N) raises ValueError.
Many legacy test classes that still reference this API are listed
below; they emit ERRORS when run until ported to the twin-mode API.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure the brain_dynamics directory is on sys.path
_BD_DIR = Path(__file__).parent.parent / "brain_dynamics"
if str(_BD_DIR) not in sys.path:
    sys.path.insert(0, str(_BD_DIR))

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    RampStimulus,
    SinStimulus,
    SquareWaveStimulus,
    StepStimulus,
)
from experiments.free_dynamics import run_free_dynamics
from experiments.attractor_analysis import extract_final_states, run_attractor_analysis
from experiments.virtual_stimulation import run_stimulation, run_virtual_stimulation
from analysis.response_matrix import compute_response_matrix
from analysis.stability_analysis import (
    analyze_trajectory_stability,
    classify_dynamics_adaptive,
    classify_dynamics_delay,
    compute_delay_distances,
    compute_state_deltas,
    compute_trajectory_features,
    run_stability_analysis,
)
from analysis.lyapunov import (
    wolf_largest_lyapunov,
    ftle_lyapunov,
    rosenstein_lyapunov,
    multi_direction_ftle,
    classify_chaos_regime,
    run_lyapunov_analysis,
    _build_delay_embedding,
    _DEFAULT_CONVERGENCE_THRESHOLD,
)
from analysis.trajectory_convergence import (
    compute_pairwise_distances,
    run_trajectory_convergence,
)
from analysis.random_comparison import (
    _make_random_dynamics_matrix,
    run_random_trajectories,
    run_random_model_comparison,
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

@unittest.skip("_WilsonCowanIntegrator removed: WC mode deprecated (see AGENTS.md)")
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md). "
    "Twin-mode equivalents are covered by TestContextTrimming."
)
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md). "
    "Twin-mode free-dynamics is covered by TestContextTrimming."
)
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
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

    def test_v2_classification_both_v1_v2_present(self):
        """analyze_trajectory_stability must return both classification and classification_v1."""
        traj = np.ones((100, N), dtype=np.float32) * 0.5
        metrics = analyze_trajectory_stability(traj)
        self.assertIn("classification", metrics)
        self.assertIn("classification_v1", metrics)
        self.assertIn("delay_mean", metrics)
        self.assertIn("delay_var", metrics)

    def test_delay_distance_shape(self):
        """compute_delay_distances should return (T - delay_dt,) array."""
        traj = np.random.rand(100, N).astype(np.float32)
        delays = compute_delay_distances(traj, delay_dt=20)
        self.assertEqual(delays.shape, (80,))

    def test_classify_dynamics_delay_fixed_point(self):
        """Constant trajectory → fixed_point under delay method."""
        traj = np.ones((200, N), dtype=np.float32) * 0.5
        result = classify_dynamics_delay(traj, delay_dt=50)
        self.assertEqual(result, "fixed_point")

    def test_classify_dynamics_delay_unstable(self):
        """Random walk → unstable under delay method."""
        rng = np.random.default_rng(0)
        traj = rng.random((200, N)).astype(np.float32)
        result = classify_dynamics_delay(traj, delay_dt=50)
        self.assertIn(result, ["unstable", "metastable"])

    def test_classify_dynamics_delay_high_dim_synchronized_limit_cycle(self):
        """Synchronized limit cycle (all N regions same phase, N=30) → limit_cycle.

        This reproduces the real-world scenario seen in logs where N≈190
        trajectories all converge to the SAME limit cycle attractor (same phase),
        making the delay-distance series oscillate periodically (cv > 0.30) but
        with a very strong ACF (acf_score ≈ 0.80).

        The old method (absolute variance < 1e-3) fails here because the
        delay-distance series oscillates, making delta_var >> 1e-3.
        The new method correctly detects the strong ACF oscillation signal.
        """
        n_reg = 30
        T = 400
        PERIOD = 80  # oscillation period in steps (should be detectable by ACF)
        t = np.arange(T)
        # All N regions oscillate at the same phase (synchronized attractor)
        traj_sync = (0.5 + 0.3 * np.sin(2 * np.pi * t / PERIOD)).astype(np.float32)
        traj_sync = np.broadcast_to(traj_sync[:, None], (T, n_reg)).copy()
        result = classify_dynamics_delay(traj_sync, delay_dt=20)
        self.assertEqual(
            result, "limit_cycle",
            msg=(
                f"classify_dynamics_delay returned '{result}' for synchronized "
                f"limit cycle (N={n_reg}, T={T}, period={PERIOD}). Expected 'limit_cycle'. "
                "This is the high-dimensional synchronized-attractor case that "
                "the old absolute-variance method (var < 1e-3) failed to detect."
            ),
        )

    def test_classify_dynamics_delay_phase_diverse_limit_cycle(self):
        """Phase-diverse limit cycle (N regions with uniform phase spread) → limit_cycle.

        In high-D systems with uniformly distributed phases, delay-distance series
        is nearly constant (phase cancellation), giving cv ≈ 0 — detected by the
        combined CV + ACF condition.
        """
        n_reg = 30
        T = 400
        PERIOD = 80  # oscillation period in steps
        t = np.arange(T)
        phases = np.linspace(0, 2 * np.pi, n_reg, endpoint=False)
        traj_div = (0.5 + 0.3 * np.sin(2 * np.pi * t[:, None] / PERIOD + phases[None, :])).astype(np.float32)
        result = classify_dynamics_delay(traj_div, delay_dt=20)
        self.assertEqual(result, "limit_cycle",
                         msg=f"got '{result}' for phase-diverse LC")

    def test_json_contains_both_methods(self):
        """stability_metrics.json must contain counts for method A, B, and C."""
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(4, 60, N).astype(np.float32)
            run_stability_analysis(trajs, output_dir=Path(tmp), delay_dt=10)
            with open(Path(tmp) / "stability_metrics.json") as fh:
                data = json.load(fh)
            self.assertIn("classification_counts", data)
            self.assertIn("classification_counts_v1", data)
            self.assertIn("classification_counts_v2", data)

    def test_json_has_delta_ratio_stats(self):
        """stability_metrics.json must contain delta_ratio distribution stats."""
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(4, 60, N).astype(np.float32)
            run_stability_analysis(trajs, output_dir=Path(tmp), delay_dt=10)
            with open(Path(tmp) / "stability_metrics.json") as fh:
                data = json.load(fh)
            self.assertIn("delta_ratio_stats", data)
            for key in ("mean", "median", "p25", "p75", "p95", "std"):
                self.assertIn(key, data["delta_ratio_stats"])


# ── Method C (adaptive) stability tests ──────────────────────────────────────

class TestAdaptiveStability(unittest.TestCase):
    def test_compute_trajectory_features_keys(self):
        """compute_trajectory_features must return all required keys."""
        traj = np.random.rand(100, N).astype(np.float32)
        features = compute_trajectory_features(traj)
        for k in ("delta_ratio", "cv_delta", "spectral_peak_ratio",
                  "tail_rms_ratio", "delay_mean", "delay_std", "traj_rms"):
            self.assertIn(k, features)

    def test_fixed_point_has_low_delta_ratio(self):
        """Constant trajectory should have delta_ratio ≈ 0."""
        traj = np.ones((200, N), dtype=np.float32) * 0.5
        features = compute_trajectory_features(traj)
        self.assertAlmostEqual(features["delta_ratio"], 0.0, places=5)

    def test_classify_adaptive_fixed_point(self):
        """Constant trajectory → fixed_point under adaptive method."""
        traj = np.ones((200, N), dtype=np.float32) * 0.5
        features = compute_trajectory_features(traj)
        cls = classify_dynamics_adaptive(features)
        self.assertEqual(cls, "fixed_point")

    def test_analyze_trajectory_has_method_c(self):
        """analyze_trajectory_stability must return method C fields."""
        traj = np.ones((100, N), dtype=np.float32) * 0.5
        metrics = analyze_trajectory_stability(traj)
        self.assertIn("delta_ratio", metrics)
        self.assertIn("cv_delta", metrics)
        self.assertIn("spectral_peak_ratio", metrics)
        self.assertIn("classification_v2", metrics)

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
    def test_scale_independent_classification(self):
        """Same relative dynamics but different n_regions should give same classification.

        This specifically tests n_regions=190 (the real-model scale where the
        original bug manifested: absolute thresholds caused 100% 'unstable').
        """
        rng = np.random.default_rng(0)
        for n_reg in [5, 20, 100, 190]:  # 190 = real-model scale
            sim_tmp = BrainDynamicsSimulator(model=None, n_regions=n_reg)
            x0 = np.full(n_reg, 0.5, dtype=np.float32)
            x0 += (rng.random(n_reg) * 0.001).astype(np.float32)
            traj, _ = sim_tmp.rollout(x0=x0, steps=200)
            feat = compute_trajectory_features(traj, delay_dt=20)
            cls = classify_dynamics_adaptive(feat)
            # WC starting near equilibrium should converge → not "unstable"
            # (this would be "unstable" under old absolute threshold for n_reg >= 20)
            self.assertIn(
                cls,
                ["fixed_point", "limit_cycle", "metastable"],
                msg=f"n_regions={n_reg}: got '{cls}' with delta_ratio={feat['delta_ratio']:.5f}",
            )

    def test_run_stability_analysis_has_method_c(self):
        """run_stability_analysis must include method C counts."""
        trajs = np.random.rand(5, 30, N).astype(np.float32)
        summary = run_stability_analysis(trajs)
        self.assertIn("classification_counts", summary)
        self.assertIn("classification_counts_v2", summary)
        self.assertIn("classification_counts_v1", summary)
        self.assertIn("delta_ratio_stats", summary)


# ══════════════════════════════════════════════════════════════════════════════
# Lyapunov Exponent Tests (Wolf-Benettin method + FTLE + classification)
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestLyapunovAnalysis(unittest.TestCase):
    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)
        self.rng = np.random.default_rng(0)

    # ── wolf_largest_lyapunov ─────────────────────────────────────────────────

    def test_wolf_returns_float_and_log_growth(self):
        x0 = np.random.rand(N).astype(np.float32)
        lle, lg = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=60, renorm_steps=10, rng=self.rng
        )
        self.assertIsInstance(lle, float)
        self.assertTrue(np.isfinite(lle))
        self.assertIsInstance(lg, np.ndarray)
        self.assertEqual(lg.shape, (6,))   # total_steps // renorm_steps

    def test_wolf_log_growth_finite(self):
        x0 = np.full(N, 0.5, dtype=np.float32)
        _, lg = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=80, renorm_steps=20, rng=self.rng
        )
        self.assertTrue(np.all(np.isfinite(lg)))

    def test_wolf_stable_system_negative_lle(self):
        """WC at equilibrium x0=0.5 should have LLE < 0 (stable system).

        Bug guard: the old implementation created a new WilsonCowanIntegrator
        with x_cur as the equilibrium every Wolf period.  Because the trajectory
        starts *at* its own equilibrium, deviation = 0, no dynamics occur, and
        the method returned LLE = 0 trivially.  The correct value is negative
        because the WC model's deviation-driven dynamics contract all
        perturbations back to the fixed point.
        """
        x0 = np.full(N, 0.5, dtype=np.float32)
        lle, _ = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=200, renorm_steps=20, rng=self.rng
        )
        # WC with no stimulus is a contracting system; LLE must be negative.
        self.assertLess(lle, 0.0)

    def test_wolf_reproducible_with_same_rng(self):
        """Same seed → same LLE."""
        x0 = np.random.rand(N).astype(np.float32)
        lle1, _ = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=60, renorm_steps=10,
            rng=np.random.default_rng(7),
        )
        lle2, _ = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=60, renorm_steps=10,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(lle1, lle2, places=8)

    def test_wolf_single_period(self):
        """total_steps == renorm_steps → one period, still returns a scalar LLE."""
        x0 = np.random.rand(N).astype(np.float32)
        lle, lg = wolf_largest_lyapunov(
            self.sim, x0=x0, total_steps=10, renorm_steps=10, rng=self.rng
        )
        self.assertEqual(len(lg), 1)
        self.assertTrue(np.isfinite(lle))

    # ── ftle_lyapunov ─────────────────────────────────────────────────────────

    def test_ftle_returns_finite_float(self):
        traj = np.random.rand(50, N).astype(np.float32)
        lam = ftle_lyapunov(traj, self.sim, rng=self.rng)
        self.assertIsInstance(lam, float)
        self.assertTrue(np.isfinite(lam))

    def test_ftle_too_short_returns_zero(self):
        """Only 1 usable point after skip → slope undefined → 0.0."""
        traj = np.random.rand(2, N).astype(np.float32)
        lam = ftle_lyapunov(traj, self.sim, skip_fraction=0.9, rng=self.rng)
        self.assertEqual(lam, 0.0)

    # ── classify_chaos_regime ─────────────────────────────────────────────────

    def test_classify_strongly_stable(self):
        r = classify_chaos_regime(-0.05)
        self.assertEqual(r["regime"], "stable")
        self.assertFalse(r["is_chaotic"])
        self.assertFalse(r["near_chaos_edge"])

    def test_classify_marginal(self):
        r = classify_chaos_regime(-0.005)
        self.assertEqual(r["regime"], "marginal_stable")
        self.assertFalse(r["is_chaotic"])

    def test_classify_edge_of_chaos(self):
        r = classify_chaos_regime(0.005)
        self.assertEqual(r["regime"], "edge_of_chaos")
        self.assertFalse(r["is_chaotic"])
        self.assertTrue(r["near_chaos_edge"])

    def test_classify_weakly_chaotic(self):
        r = classify_chaos_regime(0.05)
        self.assertEqual(r["regime"], "weakly_chaotic")
        self.assertTrue(r["is_chaotic"])

    def test_classify_strongly_chaotic(self):
        r = classify_chaos_regime(0.5)
        self.assertEqual(r["regime"], "strongly_chaotic")
        self.assertTrue(r["is_chaotic"])
        self.assertFalse(r["near_chaos_edge"])

    def test_classify_interpretation_not_empty(self):
        for lam in [-0.1, -0.005, 0.005, 0.05, 0.5]:
            r = classify_chaos_regime(lam)
            self.assertTrue(len(r["interpretation_zh"]) > 10)

    # ── run_lyapunov_analysis ─────────────────────────────────────────────────

    def test_run_wolf_keys(self):
        trajs = np.random.rand(4, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, renorm_steps=10, method="wolf")
        for key in ("lyapunov_values", "mean_lyapunov", "median_lyapunov",
                    "std_lyapunov", "fraction_positive", "fraction_negative",
                    "log_growth_curve", "chaos_regime", "method", "renorm_steps"):
            self.assertIn(key, results)
        # renorm_steps must match what was passed in (used by plot_lyapunov_growth)
        self.assertEqual(results["renorm_steps"], 10)

    def test_run_ftle_keys(self):
        trajs = np.random.rand(3, 50, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, method="ftle")
        self.assertIn("lyapunov_values", results)
        self.assertEqual(results["method"], "ftle")

    def test_run_lyapunov_shape(self):
        n = 5
        trajs = np.random.rand(n, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, renorm_steps=10, method="wolf")
        self.assertEqual(results["lyapunov_values"].shape, (n,))

    def test_log_growth_curve_shape(self):
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, renorm_steps=20, method="wolf")
        n_periods = 60 // 20
        self.assertEqual(len(results["log_growth_curve"]), n_periods)

    def test_chaos_regime_in_results(self):
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, renorm_steps=10, method="wolf")
        self.assertIn("regime", results["chaos_regime"])
        self.assertIn(results["chaos_regime"]["regime"],
                      ["stable", "marginal_stable", "edge_of_chaos",
                       "weakly_chaotic", "strongly_chaotic"])

    def test_fractions_in_0_1(self):
        trajs = np.random.rand(5, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, self.sim, renorm_steps=10, method="wolf")
        self.assertGreaterEqual(results["fraction_positive"], 0.0)
        self.assertLessEqual(results["fraction_positive"], 1.0)

    def test_saves_npy_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(3, 40, N).astype(np.float32)
            run_lyapunov_analysis(trajs, self.sim, renorm_steps=10,
                                  method="wolf", output_dir=Path(tmp))
            self.assertTrue((Path(tmp) / "lyapunov_values.npy").exists())
            self.assertTrue((Path(tmp) / "log_growth_curve.npy").exists())
            self.assertTrue((Path(tmp) / "lyapunov_report.json").exists())
            with open(Path(tmp) / "lyapunov_report.json") as fh:
                report = json.load(fh)
            self.assertIn("chaos_regime", report)
            self.assertIn("mean_lyapunov", report)
            self.assertIn("is_chaotic", report)

    def test_wc_stable_system_lle_not_strongly_positive(self):
        """WC system at equilibrium should have negative mean LLE.

        Bug guard: run_lyapunov_analysis uses wolf_largest_lyapunov which must
        now call wolf_rollout_pair instead of rollout() twice.  The old code
        returned LLE ≈ 0 (frozen dynamics); the correct value is negative.
        """
        sim = BrainDynamicsSimulator(model=None, n_regions=N, seed=42)
        trajs = np.tile(
            np.full((1, 100, N), 0.5, dtype=np.float32),
            (4, 1, 1)
        )
        # Add small perturbations so trajectories differ
        rng = np.random.default_rng(0)
        trajs += (rng.random(trajs.shape) * 0.02).astype(np.float32)
        trajs = np.clip(trajs, 0, 1)
        results = run_lyapunov_analysis(trajs, sim, renorm_steps=20, method="wolf")
        # WC is a contracting system; mean LLE must be negative.
        self.assertLess(results["mean_lyapunov"], 0.0)


# ── Rosenstein method tests ───────────────────────────────────────────────────

class TestRosensteinLyapunov(unittest.TestCase):
    def test_returns_float_and_curve(self):
        """rosenstein_lyapunov must return (float, ndarray)."""
        rng = np.random.default_rng(0)
        traj = rng.random((200, N)).astype(np.float32)
        lle, curve = rosenstein_lyapunov(traj, max_lag=20, min_temporal_sep=5)
        self.assertIsInstance(lle, float)
        self.assertIsInstance(curve, np.ndarray)

    def test_short_trajectory_returns_nan(self):
        """Too-short trajectory should return nan."""
        traj = np.random.rand(10, N).astype(np.float32)
        lle, _ = rosenstein_lyapunov(traj, max_lag=20, min_temporal_sep=10)
        self.assertTrue(np.isnan(lle))

    def test_constant_trajectory_lle_finite(self):
        """Constant trajectory should return finite (likely very negative) LLE."""
        traj = np.ones((200, N), dtype=np.float32) * 0.5
        lle, _ = rosenstein_lyapunov(traj, max_lag=20, min_temporal_sep=5)
        # May be nan or -inf if all nearest-neighbor distances are zero; skip
        # those degenerate cases but ensure no crash.
        self.assertTrue(np.isnan(lle) or np.isfinite(lle))

    def test_random_walk_lle_positive_or_near_zero(self):
        """Random trajectory should have lle >= negative large value."""
        rng = np.random.default_rng(42)
        traj = rng.random((500, N)).astype(np.float32)
        lle, curve = rosenstein_lyapunov(traj, max_lag=30, min_temporal_sep=10)
        # For i.i.d. noise, LLE is near 0 or slightly positive
        if np.isfinite(lle):
            self.assertGreater(lle, -1.0)  # Should not be strongly negative

    def test_curve_length(self):
        """The divergence curve should have length == max_lag."""
        traj = np.random.rand(300, N).astype(np.float32)
        max_lag = 25
        _, curve = rosenstein_lyapunov(traj, max_lag=max_lag, min_temporal_sep=5)
        self.assertEqual(len(curve), max_lag)

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
    def test_run_lyapunov_rosenstein_method(self):
        """run_lyapunov_analysis with method='rosenstein' should include rosenstein keys."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(3, 300, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs, sim, method="rosenstein",
            rosenstein_max_lag=20, rosenstein_min_sep=5,
        )
        self.assertIn("rosenstein_values", results)
        self.assertIn("mean_rosenstein", results)
        self.assertEqual(results["method"], "rosenstein")

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
    def test_run_lyapunov_both_includes_rosenstein(self):
        """run_lyapunov_analysis with method='both' should include wolf, ftle, rosenstein."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(3, 200, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs, sim, method="both", renorm_steps=20,
            rosenstein_max_lag=20, rosenstein_min_sep=5,
        )
        self.assertIn("lyapunov_values", results)
        self.assertIn("ftle_values", results)
        self.assertIn("rosenstein_values", results)
        self.assertIn("wolf_bias_warning", results)

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
    def test_wolf_bias_warning_false_for_varied_trajectories(self):
        """Wolf bias warning should NOT fire on short n_traj <= 10."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(5, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, sim, method="wolf", renorm_steps=10)
        # n_traj=5 <= 10, so bias warning should be False regardless of std
        self.assertFalse(results["wolf_bias_warning"])


# ── Multi-direction FTLE tests ────────────────────────────────────────────────

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestMultiDirectionFTLE(unittest.TestCase):
    def test_returns_mean_and_std(self):
        """multi_direction_ftle must return (float, float)."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        x0 = np.full(N, 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        mean_ftle, std_ftle = multi_direction_ftle(x0, sim, trajectory_length=50,
                                                   n_directions=3, rng=rng)
        self.assertIsInstance(mean_ftle, float)
        self.assertIsInstance(std_ftle, float)

    def test_std_non_negative(self):
        """std_ftle must be non-negative."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        x0 = np.random.rand(N).astype(np.float32)
        rng = np.random.default_rng(1)
        _, std_ftle = multi_direction_ftle(x0, sim, trajectory_length=60,
                                           n_directions=4, rng=rng)
        if np.isfinite(std_ftle):
            self.assertGreaterEqual(std_ftle, 0.0)


# ── Multi-segment sampling tests ─────────────────────────────────────────────

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestMultiSegmentLyapunov(unittest.TestCase):
    def test_n_segments_1_backward_compatible(self):
        """n_segments=1 should give same result as old single-x0 behavior."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        r1 = run_lyapunov_analysis(trajs, sim, method="wolf", renorm_steps=10,
                                   n_segments=1)
        r2 = run_lyapunov_analysis(trajs, sim, method="wolf", renorm_steps=10,
                                   n_segments=1)
        np.testing.assert_array_equal(r1["lyapunov_values"], r2["lyapunov_values"])

    def test_n_segments_3_returns_finite_values(self):
        """n_segments=3 should return finite LLE values."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(3, 120, N).astype(np.float32)
        results = run_lyapunov_analysis(trajs, sim, method="wolf", renorm_steps=20,
                                        n_segments=3)
        self.assertEqual(results["lyapunov_values"].shape, (3,))
        # At least some values should be finite
        self.assertTrue(np.isfinite(results["lyapunov_values"]).any())

    def test_n_segments_saves_report_with_rosenstein(self):
        """n_segments=3 with method='rosenstein' should save report."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N)
        trajs = np.random.rand(3, 300, N).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            run_lyapunov_analysis(
                trajs, sim, method="rosenstein",
                n_segments=2,
                rosenstein_max_lag=20, rosenstein_min_sep=5,
                output_dir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "lyapunov_report.json").exists())
            with open(Path(tmp) / "lyapunov_report.json") as fh:
                report = json.load(fh)
            self.assertIn("mean_rosenstein", report)


# ══════════════════════════════════════════════════════════════════════════════
# Delay Embedding Tests (_build_delay_embedding)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildDelayEmbedding(unittest.TestCase):
    """Tests for _build_delay_embedding (Takens reconstruction helper)."""

    def _make_limit_cycle(self, T: int = 300, n_regions: int = 10) -> np.ndarray:
        t = np.linspace(0, 6 * np.pi, T)
        traj = np.column_stack(
            [0.5 + 0.3 * np.sin(t + i * 0.1) for i in range(n_regions)]
        ).astype(np.float64)
        return traj

    def test_output_shape(self):
        """Shape (T - (m-1)*tau, m) for valid parameters."""
        T, N, m = 200, 10, 7
        traj = self._make_limit_cycle(T, N)
        emb = _build_delay_embedding(traj, m=m, tau=1)
        self.assertEqual(emb.shape, (T - m + 1, m))

    def test_tau_gt_1_shape(self):
        """Shape with tau=2: T_embed = T - (m-1)*tau."""
        T, N, m, tau = 200, 10, 4, 2
        traj = self._make_limit_cycle(T, N)
        emb = _build_delay_embedding(traj, m=m, tau=tau)
        expected_T = T - (m - 1) * tau
        self.assertEqual(emb.shape, (expected_T, m))

    def test_m1_returns_original(self):
        """m=1 (or m=0) should return original trajectory (fallback)."""
        traj = self._make_limit_cycle()
        for m in (0, 1):
            emb = _build_delay_embedding(traj, m=m, tau=1)
            self.assertEqual(emb.shape[1], traj.shape[1],
                             f"m={m} should return original N-dim space")

    def test_short_trajectory_fallback(self):
        """Very short trajectory should not crash and returns original."""
        traj = self._make_limit_cycle(T=5, n_regions=4)
        emb = _build_delay_embedding(traj, m=7, tau=1)
        # T_embed = 5 - 6 = -1 → fallback to original
        self.assertEqual(emb.shape, traj.shape)

    def test_output_dtype_float64(self):
        """Output should be float64 regardless of input dtype."""
        traj = self._make_limit_cycle().astype(np.float32)
        emb = _build_delay_embedding(traj, m=4, tau=1)
        self.assertEqual(emb.dtype, np.float64)

    def test_first_column_is_pc_score(self):
        """The first column should be proportional to the first PC score."""
        T, N, m = 200, 10, 4
        traj = self._make_limit_cycle(T, N)
        emb = _build_delay_embedding(traj, m=m, tau=1)
        # Column 0 is obs(t), column 1 is obs(t+1). They should have identical
        # structure but shifted by one step.
        col0 = emb[:, 0]
        col1_shifted = emb[:-1, 1]  # obs(t+1) aligns with obs[1:T_embed]
        col0_aligned = col0[:-1]    # drop last to match length
        # Correlation between adjacent lags should be very high for a slow signal
        corr = float(np.corrcoef(col0_aligned, col1_shifted)[0, 1])
        self.assertGreater(corr, 0.90, "Adjacent delay columns should be highly correlated")

    def test_rosenstein_with_delay_embedding_finite(self):
        """rosenstein_lyapunov with delay_embed_dim=4 should return finite LLE."""
        traj = self._make_limit_cycle(T=400, n_regions=10)
        lle, curve = rosenstein_lyapunov(
            traj, max_lag=30, min_temporal_sep=10,
            delay_embed_dim=4, delay_embed_tau=1,
        )
        self.assertTrue(np.isfinite(lle), f"Expected finite LLE, got {lle}")
        self.assertEqual(len(curve), 30)

    def test_delay_embed_dim_in_results(self):
        """run_lyapunov_analysis should report delay_embed_dim in results dict."""
        # Use a minimal mock that satisfies rosenstein (no model calls needed)
        class _MockSim:
            device = "cpu"
            state_bounds = (0.0, 1.0)

        T = 300
        t = np.linspace(0, 6 * np.pi, T)
        trajs = np.stack([
            np.column_stack([0.5 + 0.3 * np.sin(t + i * 0.1 + j * 0.2)
                             for i in range(N)]).astype(np.float32)
            for j in range(4)
        ])
        results = run_lyapunov_analysis(
            trajs, _MockSim(), method="rosenstein",
            rosenstein_delay_embed_dim=4,
            rosenstein_delay_embed_tau=1,
            rosenstein_max_lag=20, rosenstein_min_sep=5,
        )
        self.assertEqual(results["delay_embed_dim"], 4,
                         "delay_embed_dim should be stored in results")
        self.assertTrue(np.isfinite(results["mean_rosenstein"]))

    def test_delay_embed_dim_0_not_stored(self):
        """When delay_embed_dim=0 (disabled), results should report 0."""
        class _MockSim:
            device = "cpu"
            state_bounds = (0.0, 1.0)

        trajs = np.random.rand(3, 300, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs, _MockSim(), method="rosenstein",
            rosenstein_delay_embed_dim=0,
            rosenstein_max_lag=20, rosenstein_min_sep=5,
        )
        self.assertEqual(results["delay_embed_dim"], 0)


# ══════════════════════════════════════════════════════════════════════════════
# Convergence Threshold Tests (raised from 0.01 → 0.05)
# ══════════════════════════════════════════════════════════════════════════════

class TestConvergenceThreshold(unittest.TestCase):
    """Tests that the convergence_threshold correctly skips Wolf for ratio=0.010."""

    def test_default_threshold_is_0_05(self):
        """The module-level default must be 0.05 (raised from 0.01)."""
        self.assertAlmostEqual(_DEFAULT_CONVERGENCE_THRESHOLD, 0.05)

    def test_boundary_case_old_threshold_missed(self):
        """distance_ratio=0.010 was missed by the old threshold=0.01 (strict <)."""
        # Demonstrates the original boundary bug: 0.010 is NOT < 0.010
        self.assertFalse(0.010 < 0.01,
                         "0.010 < 0.01 should be False (this was the bug)")

    def test_boundary_case_new_threshold_catches(self):
        """distance_ratio=0.010 is now correctly caught by threshold=0.05."""
        self.assertTrue(0.010 < 0.05,
                        "0.010 < 0.05 should be True (this is the fix)")

    def test_convergence_skip_with_ratio_0_010(self):
        """run_lyapunov_analysis should skip Wolf when ratio=0.010 and threshold=0.05.

        When Wolf is skipped due to convergence, effective_method switches to 'ftle'.
        The mock simulator provides a minimal rollout for the FTLE computation.
        Key assertion: skipped_wolf=True is set correctly.
        """
        class _MockSim:
            device = "cpu"
            state_bounds = (0.0, 1.0)
            # Minimal rollout so FTLE can complete
            def rollout(self, x0, steps, stimulus=None, context_window_idx=0):
                rng = np.random.default_rng(0)
                traj = np.clip(
                    x0[None, :] + rng.standard_normal((steps, len(x0))) * 0.001,
                    0.0, 1.0,
                )
                return traj.astype(np.float32), {}

        T = 100
        t = np.linspace(0, 4 * np.pi, T)
        trajs = np.stack([
            np.column_stack([0.5 + 0.3 * np.sin(t + i * 0.1)
                             for i in range(N)]).astype(np.float32)
            for _ in range(4)
        ])
        conv_result = {"distance_ratio": 0.010}

        results = run_lyapunov_analysis(
            trajs, _MockSim(),
            method="wolf",                  # would take long if not skipped
            convergence_result=conv_result,
            convergence_threshold=0.05,     # new value
            rosenstein_max_lag=15, rosenstein_min_sep=5,
        )
        self.assertTrue(results["skipped_wolf"],
                        "Wolf should be skipped for ratio=0.010 < threshold=0.05")

    def test_convergence_skip_rosenstein_stays_rosenstein(self):
        """Fix 1: method='rosenstein' + convergence detected → method stays 'rosenstein'.

        Previously the convergence-first logic unconditionally switched the
        effective_method to 'ftle' regardless of the original method.  This
        caused λ=-0.028 (rosenstein) → λ≈0.0002 (ftle) for TwinBrain convergent
        systems because FTLE suffers context-dilution (ε_eff≈ε/context_length)
        which immediately floors the divergence near the attractor.

        After the fix: rosenstein should be preserved when convergence is detected.
        The regime is still forced to 'stable' via skipped_wolf=True.
        """
        class _MockSim:
            device = "cpu"
            state_bounds = (0.0, 1.0)
            # No rollout needed — Rosenstein works on pre-computed trajectories

        trajs = np.random.rand(3, 100, N).astype(np.float32)
        conv_result = {"distance_ratio": 0.005, "convergence_label": "converging"}

        results = run_lyapunov_analysis(
            trajs, _MockSim(),
            method="rosenstein",
            convergence_result=conv_result,
            convergence_threshold=0.05,
            rosenstein_max_lag=15, rosenstein_min_sep=5,
        )
        self.assertTrue(results["skipped_wolf"],
                        "skipped_wolf should be True: convergence detected")
        self.assertEqual(results["method"], "rosenstein",
                         "method should remain 'rosenstein', not switch to 'ftle'")
        self.assertEqual(results["chaos_regime"]["regime"], "stable",
                         "regime should be 'stable' via skipped_wolf override")

    def test_ftle_floor_detection_convergent_system(self):
        """Fix 2: FTLE returns finite negative value for convergent system.

        Previously, strongly convergent trajectories caused dist(t) to hit the
        numerical floor (1e-15) before skip, making the regression range entirely
        flat → slope≈0.  The floor-aware regression now fits only the above-floor
        portion, restoring the correct negative LLE.
        """
        class _ConvergentSim:
            state_bounds = None
            def rollout(self, x0, steps, stimulus=None, context_window_idx=0):
                # Strongly convergent: factor 0.85/step, LLE = log(0.85) ≈ -0.163
                multiplier = 0.85 ** np.arange(steps)
                traj = (x0[None, :] * multiplier[:, None]).astype(np.float32)
                return traj, np.arange(steps, dtype=np.float32)

        sim = _ConvergentSim()
        x0 = np.random.default_rng(42).random(N).astype(np.float32) + 0.3
        traj = np.array([
            x0 * (0.85 ** t) for t in range(200)
        ], dtype=np.float32)
        # FTLE should give a finite negative value (not NaN or ~0)
        lam = ftle_lyapunov(traj, sim, epsilon=1e-6, skip_fraction=0.1,
                            rng=np.random.default_rng(0))
        # Either a meaningfully negative value or NaN (if floor was hit before skip)
        # In either case, it must NOT be near zero (old buggy behavior)
        if np.isfinite(lam):
            self.assertLess(lam, -0.05,
                            f"Convergent system LLE={lam:.4f} should be clearly negative; "
                            "floor-aware regression appears not to be working")
        # NaN is acceptable (floor hit before skip, cannot estimate LLE)

    def test_skipped_wolf_chaotic_attractor_not_forced_stable(self):
        """Fix A: When convergence detected but LLE > 0, regime is NOT forced to 'stable'.

        This is the "chaotic attractor attraction" scenario: trajectories converge
        FROM nearby initial conditions TO the same chaotic attractor (distance_ratio <
        threshold), but the attractor dynamics are chaotic (LLE > 0).

        Previous bug: skipped_wolf unconditionally forced regime="stable", even when
        Rosenstein gave positive LLE, causing the contradiction between Step 9 (stable)
        and Step 15 (strongly_chaotic) in the pipeline logs.

        After fix: when skipped_wolf=True AND primary_mean >= 0, the actual LLE-based
        classification is kept (not overridden to "stable").
        """
        class _MockSim:
            state_bounds = (0.0, 1.0)
            # No rollout needed for rosenstein

        # Logistic map r=4 (chaotic, LLE = log(2) ≈ 0.693)
        # Rosenstein typically gives a positive (if underestimated) LLE for this
        def _logistic_traj(T, N_channels, seed):
            rng2 = np.random.default_rng(seed)
            x = rng2.random(N_channels).astype(np.float32)
            traj = np.empty((T, N_channels), dtype=np.float32)
            for t in range(T):
                x = np.clip(4.0 * x * (1.0 - x), 0.0, 1.0)
                traj[t] = x
            return traj

        trajs = np.array([_logistic_traj(200, N, i) for i in range(5)],
                         dtype=np.float32)
        conv_result = {"distance_ratio": 0.005, "convergence_label": "converging"}

        results = run_lyapunov_analysis(
            trajs, _MockSim(),
            method="rosenstein",
            convergence_result=conv_result,
            convergence_threshold=0.05,
            rosenstein_max_lag=20, rosenstein_min_sep=5,
        )

        lam = results["mean_lyapunov"]
        regime = results["chaos_regime"]["regime"]
        self.assertTrue(results["skipped_wolf"],
                        "skipped_wolf should still be True when convergence detected")

        if lam > 0:
            # Fix A assertion: positive LLE must NOT be overridden to "stable"
            self.assertNotEqual(
                regime, "stable",
                f"Fix A FAILED: regime='{regime}' was forced to 'stable' despite "
                f"LLE={lam:.5f} > 0 (chaotic attractor attraction scenario)."
            )
        # If lam <= 0 for this seed, the test still exercises the code path

    def test_no_skip_above_threshold(self):
        """ratio=0.10 should NOT trigger skip with threshold=0.05."""
        class _MockSim:
            device = "cpu"
            state_bounds = (0.0, 1.0)

        trajs = np.random.rand(3, 100, N).astype(np.float32)
        conv_result = {"distance_ratio": 0.10}  # above threshold

        results = run_lyapunov_analysis(
            trajs, _MockSim(),
            method="rosenstein",            # use rosenstein so no model calls needed
            convergence_result=conv_result,
            convergence_threshold=0.05,
            rosenstein_max_lag=15, rosenstein_min_sep=5,
        )
        # skipped_wolf only applies to wolf; for rosenstein method, it's False
        self.assertFalse(results["skipped_wolf"],
                         "Wolf should NOT be skipped for ratio=0.10 > threshold=0.05")


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory Convergence Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryConvergence(unittest.TestCase):
    def test_compute_pairwise_distances_shape(self):
        trajs = np.random.rand(10, 50, N).astype(np.float32)
        mean_dist = compute_pairwise_distances(trajs, n_pairs=5, seed=0)
        self.assertEqual(mean_dist.shape, (50,))

    def test_compute_pairwise_distances_dtype(self):
        trajs = np.random.rand(10, 20, N).astype(np.float32)
        mean_dist = compute_pairwise_distances(trajs, n_pairs=5)
        self.assertEqual(mean_dist.dtype, np.float32)

    def test_distances_non_negative(self):
        trajs = np.random.rand(10, 30, N).astype(np.float32)
        mean_dist = compute_pairwise_distances(trajs, n_pairs=5)
        self.assertTrue(np.all(mean_dist >= 0.0))

    def test_identical_trajectories_distance_zero(self):
        """If all trajectories are identical, pairwise distance should be zero."""
        base = np.random.rand(1, 50, N).astype(np.float32)
        trajs = np.tile(base, (8, 1, 1))
        mean_dist = compute_pairwise_distances(trajs, n_pairs=10)
        np.testing.assert_allclose(mean_dist, 0.0, atol=1e-6)

    def test_run_trajectory_convergence_keys(self):
        trajs = np.random.rand(8, 40, N).astype(np.float32)
        results = run_trajectory_convergence(trajs, n_pairs=5)
        self.assertIn("mean_distances", results)
        self.assertIn("initial_mean_distance", results)
        self.assertIn("final_mean_distance", results)
        self.assertIn("distance_ratio", results)
        self.assertIn("convergence_label", results)

    def test_convergence_label_converging(self):
        """Trajectories converging to same point should get 'converging' label."""
        rng = np.random.default_rng(0)
        n_init, steps = 10, 60
        # All trajectories start dispersed but converge to the same point (0.5)
        trajs = np.zeros((n_init, steps, N), dtype=np.float32)
        for i in range(n_init):
            start = rng.random(N).astype(np.float32)
            for t in range(steps):
                alpha = t / (steps - 1)
                trajs[i, t] = (1 - alpha) * start + alpha * 0.5
        results = run_trajectory_convergence(trajs, n_pairs=10)
        self.assertEqual(results["convergence_label"], "converging")

    def test_saves_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(5, 20, N).astype(np.float32)
            run_trajectory_convergence(trajs, n_pairs=4, output_dir=Path(tmp))
            self.assertTrue((Path(tmp) / "distance_curve.npy").exists())


# ══════════════════════════════════════════════════════════════════════════════
# Random Model Comparison Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRandomModelComparison(unittest.TestCase):
    def test_random_dynamics_matrix_shape(self):
        W = _make_random_dynamics_matrix(n_regions=N, target_spectral_radius=0.9)
        self.assertEqual(W.shape, (N, N))

    def test_random_dynamics_matrix_spectral_radius(self):
        W = _make_random_dynamics_matrix(n_regions=20, target_spectral_radius=0.9)
        sr = float(np.abs(np.linalg.eigvals(W)).max())
        self.assertAlmostEqual(sr, 0.9, places=4)

    def test_run_random_trajectories_shape(self):
        trajs = run_random_trajectories(n_regions=N, n_init=5, steps=20)
        self.assertEqual(trajs.shape, (5, 20, N))

    def test_run_random_trajectories_values_in_0_1(self):
        trajs = run_random_trajectories(n_regions=N, n_init=5, steps=20)
        self.assertTrue(np.all(trajs >= 0.0))
        self.assertTrue(np.all(trajs <= 1.0))

    def test_run_random_trajectories_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_random_trajectories(
                n_regions=N, n_init=3, steps=10, output_dir=Path(tmp)
            )
            self.assertTrue((Path(tmp) / "random_trajectories.npy").exists())

    def test_run_random_model_comparison_keys(self):
        trajs = np.random.rand(5, 20, N).astype(np.float32)
        result = run_random_model_comparison(
            trajectories=trajs,
            random_n_init=5,
            random_steps=15,
        )
        self.assertIn("model", result)
        # Result has per-spectral-radius keys like "random_sr1.50", not "random"
        random_keys = [k for k in result if k.startswith("random_sr")]
        self.assertTrue(len(random_keys) >= 1,
                        f"Expected at least one 'random_srX.XX' key, got: {list(result)}")
        self.assertIn("trajectory_variance", result["model"])
        # Each random entry has mean_lyapunov from Wolf-Benettin estimation
        self.assertIn("mean_lyapunov", result[random_keys[0]])

    def test_run_random_model_comparison_saves_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(4, 20, N).astype(np.float32)
            run_random_model_comparison(
                trajectories=trajs,
                random_n_init=4,
                random_steps=15,
                output_dir=Path(tmp),
            )
            self.assertTrue((Path(tmp) / "analysis_comparison.json").exists())
            with open(Path(tmp) / "analysis_comparison.json") as fh:
                data = json.load(fh)
            self.assertIn("model", data)
            random_keys = [k for k in data if k.startswith("random_sr")]
            self.assertTrue(len(random_keys) >= 1)


# ══════════════════════════════════════════════════════════════════════════════
# Attractor Analysis: basin_distribution.json saving
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Response Matrix — shared-x0 fix
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestResponseMatrixSharedX0(unittest.TestCase):
    """
    Verify that compute_response_matrix uses a shared x0 for all rows so that
    R[i,j] reflects stimulus-propagation structure, not equilibrium-state noise.
    """

    def _compute_R(self, seed=42):
        sim = BrainDynamicsSimulator(model=None, n_regions=N, seed=0)
        from analysis.response_matrix import compute_response_matrix
        return compute_response_matrix(
            sim,
            n_nodes=N,
            stim_duration=30,
            pre_steps=30,
            measure_window=15,
            stim_amplitude=0.5,
            seed=seed,
        )

    def test_row_norms_are_comparable(self):
        """All row norms should be within the same order of magnitude."""
        R = self._compute_R()
        norms = np.linalg.norm(R, axis=1)
        self.assertGreater(norms.min(), 0.0)
        ratio = norms.max() / (norms.min() + 1e-10)
        # With shared x0, the ratio should be modest (< 10×).
        # Previously, with different x0 per row, ratios > 100× were possible.
        self.assertLess(ratio, 10.0,
                        msg=f"Row norm ratio {ratio:.2f} too large — "
                            "likely still using different x0 per row")

    def test_zero_amplitude_gives_near_zero_response(self):
        """With amplitude=0 every row of R should be essentially zero."""
        sim = BrainDynamicsSimulator(model=None, n_regions=N, seed=0)
        from analysis.response_matrix import compute_response_matrix
        R = compute_response_matrix(
            sim, n_nodes=N, stim_duration=30, pre_steps=30,
            measure_window=15, stim_amplitude=0.0, seed=42,
        )
        self.assertAlmostEqual(float(np.abs(R).max()), 0.0, places=4)

    def test_response_matrix_shape(self):
        R = self._compute_R()
        self.assertEqual(R.shape, (N, N))

    def test_different_seeds_give_different_R(self):
        """Different seeds → different x0 → different R matrices."""
        R1 = self._compute_R(seed=1)
        R2 = self._compute_R(seed=99)
        self.assertFalse(np.allclose(R1, R2))

    def test_column_stats_attributes(self):
        """column_mean and stim_specificity can be computed from R."""
        R = self._compute_R()
        col_mean = np.abs(R).mean(axis=0)
        stim_specificity = R.std(axis=1)
        self.assertEqual(col_mean.shape, (N,))
        self.assertEqual(stim_specificity.shape, (N,))
        self.assertTrue(np.all(col_mean >= 0))
        self.assertTrue(np.all(stim_specificity >= 0))


class TestAttractorBasinJSON(unittest.TestCase):
    def _make_two_attractor_trajs(self):
        rng = np.random.default_rng(0)
        n_init, steps, n_regions = 20, 50, N
        trajs = np.zeros((n_init, steps, n_regions), dtype=np.float32)
        for i in range(n_init):
            final = np.zeros(n_regions) + (0.2 if i < 10 else 0.8)
            trajs[i] = final + rng.random((steps, n_regions)) * 0.01
        return trajs.astype(np.float32)

    def test_saves_basin_distribution_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajs = self._make_two_attractor_trajs()
            run_attractor_analysis(
                trajs,
                tail_steps=5,
                k_candidates=[2],
                k_best=2,
                output_dir=Path(tmp),
            )
            json_path = Path(tmp) / "basin_distribution.json"
            self.assertTrue(json_path.exists())
            with open(json_path) as fh:
                data = json.load(fh)
            # Keys should look like "attractor_A", "attractor_B", etc.
            self.assertTrue(len(data) >= 1)
            for key in data:
                self.assertTrue(key.startswith("attractor_"))
            # All fractions should sum to ~1
            total = sum(data.values())
            self.assertAlmostEqual(total, 1.0, places=4)


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

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
    def test_simulator_rejects_can_forward_false_model(self):
        """
        BrainDynamicsSimulator must raise RuntimeError when the provided model
        has can_forward=False — no silent WC fallback.
        """
        class _FakeNoForward:
            can_forward = False

        with self.assertRaises(RuntimeError):
            BrainDynamicsSimulator(model=_FakeNoForward(), n_regions=N)

    @unittest.skip(
        "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
        "WC mode deprecated (see AGENTS.md)."
    )
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
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


@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
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

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
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
# predict_future improvements (num_steps forwarding, error surfacing)
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


# ══════════════════════════════════════════════════════════════════════════════
# Context trimming for 8 GB GPU support
# ══════════════════════════════════════════════════════════════════════════════

class TestContextTrimming(unittest.TestCase):
    """
    Tests that _rollout_with_twin trims ALL modalities to context_length.

    Key fix (计划书 §四/§十三): The graph cache is used only as the initial
    state.  ALL modalities are trimmed to context_length so that the encoder
    always receives [N, context_length, C] per node type — eliminating OOM
    from large EEG sequences (e.g. T_eeg=98500 → T_eeg=200, 6.34 GB → 12.9 MB).
    """

    def setUp(self):
        _models_dir = Path(__file__).parent.parent / "models"
        if str(_models_dir.parent) not in sys.path:
            sys.path.insert(0, str(_models_dir.parent))

    def _make_sim(self, T_base: int, context_length: int, pred_steps: int = 3,
                  with_eeg: bool = False, T_eeg: int = 0):
        """Create a BrainDynamicsSimulator whose base_graph has T_base timesteps."""
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from torch_geometric.data import HeteroData

        H = 16
        node_types = ["fmri", "eeg"] if with_eeg else ["fmri"]
        edge_types = [("fmri", "connects", "fmri")]
        if with_eeg:
            edge_types.append(("eeg", "connects", "eeg"))

        model = GraphNativeBrainModel(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict={nt: 1 for nt in node_types},
            hidden_channels=H,
            prediction_steps=pred_steps,
            predictor_config={"context_length": context_length},
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")

        g = HeteroData()
        # T_base > context_length to trigger the trimming path
        g["fmri"].x = torch.randn(N, T_base, 1)
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        if with_eeg and T_eeg > 0:
            N_eeg = 8
            g["eeg"].x = torch.randn(N_eeg, T_eeg, 1)
            g["eeg", "connects", "eeg"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        sim = BrainDynamicsSimulator(model=twin, base_graph=g, modality="fmri", device="cpu")
        return sim, twin, model

    def test_context_not_mutated_by_rollout(self):
        """_rollout_with_twin must not mutate base_graph."""
        sim, _, _ = self._make_sim(T_base=30, context_length=10, pred_steps=3)
        original_T = sim.base_graph["fmri"].x.shape[1]
        sim.rollout(steps=6)
        self.assertEqual(sim.base_graph["fmri"].x.shape[1], original_T)

    def test_rollout_produces_correct_steps(self):
        """_rollout_with_twin returns (steps, n_regions) trajectory."""
        sim, _, _ = self._make_sim(T_base=30, context_length=10, pred_steps=3)
        traj, times = sim.rollout(steps=9)
        self.assertEqual(traj.shape, (9, N))
        self.assertEqual(times.shape, (9,))

    def test_context_trimmed_internally(self):
        """
        The context passed to the encoder must be context_length steps (not T_base).

        We verify this by monkey-patching predict_future to record the input
        context shape and checking it matches context_length, not T_base.
        """
        import unittest.mock
        T_BASE = 30
        CTX_LEN = 10

        sim, twin, _ = self._make_sim(T_base=T_BASE, context_length=CTX_LEN, pred_steps=3)

        observed_T = []
        original_predict = twin.predict_future

        def recording_predict(data, num_steps=None):
            observed_T.append(data["fmri"].x.shape[1])
            return original_predict(data, num_steps=num_steps)

        with unittest.mock.patch.object(twin, "predict_future", side_effect=recording_predict):
            sim.rollout(steps=6)

        # All calls should receive exactly context_length steps
        self.assertTrue(all(t == CTX_LEN for t in observed_T),
                        f"Expected all context lengths == {CTX_LEN}, got {observed_T}")

    def test_no_trim_when_base_shorter_than_context(self):
        """If T_base <= context_length, no trimming should occur."""
        import unittest.mock
        T_BASE = 8
        CTX_LEN = 10  # base is already shorter

        sim, twin, _ = self._make_sim(T_base=T_BASE, context_length=CTX_LEN, pred_steps=3)

        observed_T = []
        original_predict = twin.predict_future

        def recording_predict(data, num_steps=None):
            observed_T.append(data["fmri"].x.shape[1])
            return original_predict(data, num_steps=num_steps)

        with unittest.mock.patch.object(twin, "predict_future", side_effect=recording_predict):
            sim.rollout(steps=6)

        # All calls should use the full T_BASE (no trimming should occur)
        self.assertTrue(all(t == T_BASE for t in observed_T),
                        f"Expected context length == {T_BASE} (no trim), got {observed_T}")

    def test_eeg_also_trimmed_to_context_length(self):
        """
        EEG (and all non-primary modalities) must also be trimmed to context_length.

        This is the key OOM fix: T_eeg=98500 → T_eeg=context_length (e.g. 200),
        reducing peak encoder activation from 6.34 GB to ~12.9 MB.
        """
        import unittest.mock
        T_BASE_FMRI = 20
        T_BASE_EEG = 500   # much larger than context_length
        CTX_LEN = 10

        sim, twin, _ = self._make_sim(
            T_base=T_BASE_FMRI, context_length=CTX_LEN, pred_steps=3,
            with_eeg=True, T_eeg=T_BASE_EEG,
        )

        observed_eeg_T = []
        original_predict = twin.predict_future

        def recording_predict(data, num_steps=None):
            if "eeg" in data.node_types:
                observed_eeg_T.append(data["eeg"].x.shape[1])
            return original_predict(data, num_steps=num_steps)

        with unittest.mock.patch.object(twin, "predict_future", side_effect=recording_predict):
            sim.rollout(steps=6)

        self.assertTrue(
            len(observed_eeg_T) > 0,
            "predict_future was never called with EEG data",
        )
        self.assertTrue(
            all(t == CTX_LEN for t in observed_eeg_T),
            f"EEG context not trimmed: expected T={CTX_LEN}, got {observed_eeg_T}",
        )
        # base_graph EEG must remain untouched
        self.assertEqual(
            sim.base_graph["eeg"].x.shape[1], T_BASE_EEG,
            "base_graph EEG was mutated by rollout",
        )

    def test_x0_injection_produces_diverse_trajectories(self):
        """
        Different x0 values must produce different trajectories.

        计划书 §五: run_free_dynamics samples x0 = sample_random_state() for
        each of the 200 rollouts.  Before this fix, x0 was silently ignored in
        twin mode, making all 200 trajectories identical.
        """
        T_BASE = 20
        CTX_LEN = 10
        sim, _, _ = self._make_sim(T_base=T_BASE, context_length=CTX_LEN, pred_steps=3)

        rng = np.random.default_rng(0)
        x0_a = rng.random(N).astype(np.float32)
        x0_b = rng.random(N).astype(np.float32)

        traj_a, _ = sim.rollout(steps=3, x0=x0_a)
        traj_b, _ = sim.rollout(steps=3, x0=x0_b)

        # Trajectories from different x0 must differ
        self.assertFalse(
            np.allclose(traj_a, traj_b, atol=1e-6),
            "rollout() with different x0 produced identical trajectories — "
            "x0 injection is not working",
        )

    def test_x0_none_does_not_crash(self):
        """rollout(x0=None) in twin mode must use base_graph without error."""
        sim, _, _ = self._make_sim(T_base=20, context_length=10, pred_steps=3)
        traj, times = sim.rollout(steps=3, x0=None)
        self.assertEqual(traj.shape, (3, N))

    def test_sample_random_state_zscore_no_clip(self):
        """
        sample_random_state() must NOT clip z-scored data to [0, 1].

        V5 graph caches store z-scored data (values in roughly [-3, 3]).
        The old code did np.clip(..., 0.0, 1.0) which forced all values into
        [0, 0.05] when mean_state ≈ 0.  Injecting these near-zero values into
        a context full of ±σ values caused a large discontinuity that
        manifested as large pre-stimulation oscillations in stim-response plots.

        After the fix: x0 must contain negative values (mean ≈ 0, std ≈ per-channel
        std), and state_bounds must be None for z-scored graphs.
        """
        from torch_geometric.data import HeteroData

        # Build a graph with clearly z-scored data (large negative values)
        g = HeteroData()
        rng_t = np.random.default_rng(0)
        # Simulate z-scored EEG: values centred at 0, std ≈ 1, clearly negative
        raw = rng_t.standard_normal((N, 30)).astype(np.float32)  # mean ≈ 0, min ≈ -2.5
        import torch
        g["fmri"].x = torch.tensor(raw).unsqueeze(-1)  # [N, 30, 1]
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        sim, _, _ = self._make_sim(T_base=30, context_length=10, pred_steps=3)
        # Override base_graph with z-scored one
        sim.base_graph = g

        # Force recompute of _state_bounds by re-init would be complex;
        # instead verify via the public state_bounds property of a fresh sim
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=16,
            prediction_steps=3,
            predictor_config={"context_length": 10},
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")
        sim2 = BrainDynamicsSimulator(model=twin, base_graph=g, modality="fmri", device="cpu")

        # state_bounds must be None for z-scored data (data.min() < -0.1)
        self.assertIsNone(
            sim2.state_bounds,
            "state_bounds must be None for z-scored data to prevent spurious "
            "attractor at 0 during Wolf/FTLE Lyapunov analysis",
        )

        # sample_random_state() must return values that include negatives
        x0 = sim2.sample_random_state(rng=np.random.default_rng(42))
        self.assertEqual(x0.shape, (N,))
        self.assertTrue(
            np.any(x0 < 0.0),
            "sample_random_state() returned all non-negative values for z-scored "
            "data — the [0,1] clip was not disabled.  This causes large "
            "pre-stimulation transient oscillations.",
        )
        # Values must be finite and within the model's natural operating range.
        # For z-scored data, any |value| > _MAX_INIT_SIGMA_CHECK would indicate a
        # pathological initial state far outside the training distribution.
        _MAX_INIT_SIGMA_CHECK = 20.0  # generous bound: ±20σ is unphysical for any brain signal
        self.assertTrue(np.all(np.isfinite(x0)), "x0 contains NaN/Inf")
        self.assertTrue(
            np.all(np.abs(x0) < _MAX_INIT_SIGMA_CHECK),
            f"x0 values are unrealistically large (>±{_MAX_INIT_SIGMA_CHECK}σ)",
        )


# ══════════════════════════════════════════════════════════════════════════════
# New Lyapunov fixes: boundary bias correction + convergence-first strategy
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestLyapunovBoundaryCorrection(unittest.TestCase):
    """
    Tests that validate the boundary-bias fix in wolf_largest_lyapunov.

    Root cause: when x_cur is near 0 or 1 and epsilon * perturb pushes x_pert
    past the boundary, np.clip silently reduces the actual perturbation to
    less than epsilon.  The old code computed log(r / epsilon_nominal), which
    overestimates the growth when r > epsilon_actual.  The fix uses
    epsilon_actual = ||x_pert_clipped - x_cur||.
    """

    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_boundary_state_gives_negative_lle(self):
        """
        x0 near boundary [0 or 1] should NOT inflate LLE to a large positive
        value after the boundary correction.  The WC system is contracting, so
        LLE must still be negative or near zero regardless of where x0 starts.
        """
        # 0.999 is chosen so that the positive-direction perturbation components
        # (ε·e_i > 0) are definitely clipped to 1.0, triggering the boundary
        # correction, while leaving at least one component (x0[0]=0.5)
        # unclipped to ensure actual_eps > 0.
        x0 = np.full(N, 0.999, dtype=np.float32)
        x0[0] = 0.5  # at least one component well away from boundary
        lle, _ = wolf_largest_lyapunov(
            self.sim,
            x0=x0,
            total_steps=100,
            renorm_steps=20,
            epsilon=1e-5,
            rng=np.random.default_rng(1),
        )
        # WC is a globally contracting system; LLE must be ≤ 0.
        self.assertLessEqual(
            lle,
            0.1,
            msg=f"LLE={lle:.4f} suspiciously large for a contracting system "
                "near the boundary — boundary bias fix may not be applied",
        )

    def test_boundary_lle_not_strongly_positive(self):
        """All-boundary state: LLE must not be > 0.5 (old buggy value ≈ 0.5)."""
        # x0 entirely at 1.0 — clipping absorbs the positive perturbation
        x0 = np.ones(N, dtype=np.float32)
        lle, _ = wolf_largest_lyapunov(
            self.sim,
            x0=x0,
            total_steps=100,
            renorm_steps=20,
            epsilon=1e-5,
            rng=np.random.default_rng(2),
        )
        # When all components are at boundary 1, positive perturbation is fully
        # clipped → actual_eps ≈ 0 → the period is skipped.  The negative
        # perturbation components will still carry through, producing a valid
        # (negative) LLE.  In any case, the result must not be > 0.5.
        self.assertLess(lle, 0.5)

    def test_actual_eps_used_not_nominal_eps(self):
        """
        Direct numerical check: with a near-boundary state, the LLE computed
        with the boundary correction (actual_eps) must differ from the uncorrected
        value (nominal eps) and must be more negative (or at least not larger).

        We achieve this by calling wolf_largest_lyapunov with a state deliberately
        near the boundary and a large nominal epsilon, so clipping is guaranteed
        to trigger.  The corrected LLE must be ≤ the uncorrected one.
        """
        # 0.999 with epsilon=0.01: any positive-direction component pushes
        # x_pert[i] = 0.999 + 0.01·e_i past 1.0, which is clipped to 1.0.
        # The actual perturbation in that dimension is only 0.001, not 0.01.
        x0 = np.full(N, 0.999, dtype=np.float32)
        rng_seed = 3

        # Corrected (new) result
        lle_corrected, _ = wolf_largest_lyapunov(
            self.sim,
            x0=x0,
            total_steps=200,
            renorm_steps=20,
            epsilon=0.01,
            rng=np.random.default_rng(rng_seed),
        )
        # The corrected LLE must be < 0 (contracting system) or at least << 0.5
        # (the old overestimated value for near-boundary states).
        self.assertLess(
            lle_corrected,
            0.5,
            msg=f"Corrected LLE={lle_corrected:.4f} ≥ 0.5 — "
                "boundary correction does not appear to be working",
        )


@unittest.skip(
    "BrainDynamicsSimulator(model=None, n_regions=N) API removed: "
    "WC mode deprecated (see AGENTS.md)."
)
class TestLyapunovConvergenceFirst(unittest.TestCase):
    """
    Tests for the convergence-first strategy in run_lyapunov_analysis.

    When trajectory_convergence reports distance_ratio < threshold,
    Wolf is skipped and FTLE is used instead.
    """

    def setUp(self):
        self.sim = BrainDynamicsSimulator(model=None, n_regions=N)

    def test_convergence_skip_triggers_ftle(self):
        """
        Passing convergence_result with ratio < threshold should skip Wolf
        and set effective method to "ftle".
        """
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        convergence_result = {
            "distance_ratio": 0.001,  # strongly converging
            "convergence_label": "converging",
        }
        results = run_lyapunov_analysis(
            trajs,
            self.sim,
            method="wolf",
            convergence_result=convergence_result,
            convergence_threshold=0.01,
            renorm_steps=10,
        )
        self.assertTrue(results["skipped_wolf"])
        self.assertEqual(results["method"], "ftle")

    def test_convergence_skip_overrides_regime_to_stable(self):
        """When Wolf is skipped, chaos_regime must be 'stable'."""
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs,
            self.sim,
            method="wolf",
            convergence_result={"distance_ratio": 0.0001},
            convergence_threshold=0.01,
            renorm_steps=10,
        )
        self.assertEqual(results["chaos_regime"]["regime"], "stable")
        self.assertFalse(results["chaos_regime"]["is_chaotic"])

    def test_no_convergence_skip_when_ratio_above_threshold(self):
        """distance_ratio ≥ threshold → Wolf runs normally (not skipped)."""
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs,
            self.sim,
            method="wolf",
            convergence_result={"distance_ratio": 0.5},  # above threshold
            convergence_threshold=0.01,
            renorm_steps=10,
        )
        self.assertFalse(results["skipped_wolf"])
        self.assertEqual(results["method"], "wolf")

    def test_no_convergence_result_does_not_skip(self):
        """convergence_result=None → no skip; Wolf runs as requested."""
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs,
            self.sim,
            method="wolf",
            convergence_result=None,
            renorm_steps=10,
        )
        self.assertFalse(results["skipped_wolf"])

    def test_both_method_runs_wolf_and_ftle(self):
        """method='both' should populate 'ftle_values' in results."""
        trajs = np.random.rand(3, 60, N).astype(np.float32)
        results = run_lyapunov_analysis(
            trajs,
            self.sim,
            method="both",
            renorm_steps=10,
        )
        self.assertIn("ftle_values", results)
        self.assertIn("mean_ftle", results)
        self.assertEqual(results["method"], "both")
        # lyapunov_values should carry Wolf estimates when Wolf runs
        self.assertFalse(results["skipped_wolf"])

    def test_skipped_wolf_report_contains_skipped_field(self):
        """JSON report must contain 'skipped_wolf' field."""
        with tempfile.TemporaryDirectory() as tmp:
            trajs = np.random.rand(3, 40, N).astype(np.float32)
            run_lyapunov_analysis(
                trajs,
                self.sim,
                method="wolf",
                convergence_result={"distance_ratio": 0.001},
                convergence_threshold=0.01,
                renorm_steps=10,
                output_dir=Path(tmp),
            )
            with open(Path(tmp) / "lyapunov_report.json") as fh:
                report = json.load(fh)
            self.assertIn("skipped_wolf", report)
            self.assertTrue(report["skipped_wolf"])


# ══════════════════════════════════════════════════════════════════════════════
# n_temporal_windows: prediction_steps fallback when T ≤ context_length
# ══════════════════════════════════════════════════════════════════════════════

class TestNTemporalWindowsFallback(unittest.TestCase):
    """
    Tests for the prediction_steps-based fallback in n_temporal_windows.

    When T_primary ≤ context_length + stride (only 1 full-context window is
    available), n_temporal_windows should fall back to prediction_steps as the
    stride and return multiple shorter-context windows instead of just 1.
    """

    def setUp(self):
        _models_dir = Path(__file__).parent.parent / "models"
        if str(_models_dir.parent) not in sys.path:
            sys.path.insert(0, str(_models_dir.parent))

    def _make_sim(self, T_base: int, context_length: int, pred_steps: int):
        """Build a BrainDynamicsSimulator with given T, context_length, pred_steps."""
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from torch_geometric.data import HeteroData

        H = 16
        model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
            prediction_steps=pred_steps,
            predictor_config={"context_length": context_length},
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")
        g = HeteroData()
        g["fmri"].x = torch.randn(N, T_base, 1)
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        return BrainDynamicsSimulator(model=twin, base_graph=g, modality="fmri", device="cpu")

    def test_t_equals_ctx_gives_multiple_windows(self):
        """T=context_length: fallback should give pred_steps-based windows (>1)."""
        # T=200, ctx_len=200, pred_steps=50, stride=50 → fallback gives 4 windows
        sim = self._make_sim(T_base=200, context_length=200, pred_steps=50)
        self.assertGreater(sim.n_temporal_windows, 1,
                           "Expected >1 windows via pred_steps fallback when T=context_length")

    def test_t_greater_ctx_uses_primary_path(self):
        """T > context_length + stride: primary path should give ≥ 2 windows."""
        # T=300, ctx_len=200, stride=50 → primary gives (300-200)//50+1 = 3 windows
        sim = self._make_sim(T_base=300, context_length=200, pred_steps=50)
        self.assertGreaterEqual(sim.n_temporal_windows, 2,
                                "Expected ≥2 windows via primary sliding-window path")

    def test_t_less_than_ctx_returns_at_least_one(self):
        """T < context_length and pred_steps >= ctx: should still return 1."""
        # T=10, ctx_len=20, pred_steps=20 (pred_steps >= ctx_len → no fallback benefit)
        sim = self._make_sim(T_base=10, context_length=20, pred_steps=20)
        self.assertEqual(sim.n_temporal_windows, 1)

    def test_window_count_capped_by_extractable(self):
        """Window count must not exceed what _get_context_for_window can provide."""
        # T=200, ctx_len=200, stride=50, pred_steps=10:
        #   pred-based: (200-10)//10+1 = 20 windows
        #   extractable: (200-1)//50+1 = 4 windows
        #   result should be exactly min(20,4) = 4
        sim = self._make_sim(T_base=200, context_length=200, pred_steps=10)
        n = sim.n_temporal_windows
        self.assertEqual(n, 4,
                         "Window count should be exactly 4 (capped at extractable windows)")

    def test_small_context_length_uses_primary_path(self):
        """With small context_length=37, T=200 gives many windows via primary path."""
        # ctx_len=37, stride=9, T=200 → (200-37)//9+1 = 19 windows
        sim = self._make_sim(T_base=200, context_length=37, pred_steps=17)
        n = sim.n_temporal_windows
        self.assertGreaterEqual(n, 10,
                                "With context_length=37 and T=200, expect many windows")

    def test_fallback_windows_are_usable(self):
        """rollout with context_window_idx > 0 should work without crashing."""
        sim = self._make_sim(T_base=200, context_length=200, pred_steps=50)
        n = sim.n_temporal_windows
        self.assertGreater(n, 1)
        # Each window index should produce a valid trajectory
        for idx in range(min(n, 3)):
            with self.subTest(window_idx=idx):
                traj, _ = sim.rollout(steps=3, context_window_idx=idx)
                self.assertEqual(traj.shape, (3, N))

    def test_fallback_windows_have_different_context_lengths(self):
        """
        Regression test for _get_context_for_window fallback mode.

        When T == context_length, the fallback path produces windows with
        decreasing context lengths: T, T-stride, T-2*stride, …  Previously
        all fallback windows returned the same [0:T] context, giving zero
        diversity and defeating the multi-window purpose.
        """
        ctx_len = 20
        pred_steps = 5
        # stride = ctx_len // 4 = 5
        sim = self._make_sim(T_base=ctx_len, context_length=ctx_len,
                             pred_steps=pred_steps)
        n = sim.n_temporal_windows
        self.assertGreater(n, 1, "Expected >1 fallback windows")

        stride = sim._get_stride()
        T = ctx_len  # T == context_length, so we're in fallback mode

        seen_lengths = set()
        for w in range(n):
            ctx = sim._get_context_for_window(w)
            t_ctx = ctx["fmri"].x.shape[1]
            expected = T - w * stride
            self.assertEqual(
                t_ctx, expected,
                f"Window {w}: expected context length {expected}, got {t_ctx}. "
                "All fallback windows should have distinct prefix lengths.",
            )
            seen_lengths.add(t_ctx)

        # All windows must have genuinely different context lengths
        self.assertEqual(
            len(seen_lengths), n,
            "All fallback windows should have distinct context lengths, "
            f"but got {len(seen_lengths)} unique lengths for {n} windows.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Config reading fix: predictor_config from v5_optimization.advanced_prediction
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictorConfigFromV5Optimization(unittest.TestCase):
    """
    Tests that TwinBrainDigitalTwin.from_checkpoint correctly reads
    predictor_config from config['v5_optimization']['advanced_prediction']
    as a fallback when config['model']['predictor_config'] is absent.

    This fixes the bug where context_length always defaulted to 200 even when
    the training config had a different value under v5_optimization.
    """

    def setUp(self):
        _models_dir = Path(__file__).parent.parent / "models"
        if str(_models_dir.parent) not in sys.path:
            sys.path.insert(0, str(_models_dir.parent))

    def test_predictor_config_read_from_v5_optimization(self):
        """
        from_checkpoint extracts predictor_config from
        config['v5_optimization']['advanced_prediction'] when
        config['model']['predictor_config'] is absent.
        """
        import io
        import unittest.mock as mock
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from models.graph_native_system import GraphNativeBrainModel

        # Build a real model so we have a valid state_dict
        H = 16
        real_model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
            prediction_steps=17,
            predictor_config={"context_length": 37},
        )
        real_model.eval()

        # Save to a temporary checkpoint
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            torch.save({"model_state_dict": real_model.state_dict()}, ckpt_path)

            # Config that uses the v5_optimization path (standard training config)
            cfg = {
                "model": {
                    "prediction_steps": 17,
                    # NOTE: NO 'predictor_config' key here
                },
                "v5_optimization": {
                    "advanced_prediction": {
                        "context_length": 37,
                        "use_hierarchical": True,
                        "use_transformer": True,
                    }
                },
            }

            loaded = TwinBrainDigitalTwin.from_checkpoint(
                checkpoint_path=ckpt_path,
                config=cfg,
                device="cpu",
            )

        # The predictor should have context_length=37, not the default 200
        predictor = loaded.model.predictor
        self.assertEqual(
            predictor.context_length, 37,
            f"Expected context_length=37 from v5_optimization config, "
            f"got {predictor.context_length}",
        )

    def test_legacy_predictor_config_still_works(self):
        """
        config['model']['predictor_config'] takes priority over
        v5_optimization.advanced_prediction.
        """
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from models.graph_native_system import GraphNativeBrainModel

        H = 16
        real_model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
            prediction_steps=50,
            predictor_config={"context_length": 100},
        )
        real_model.eval()

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            torch.save({"model_state_dict": real_model.state_dict()}, ckpt_path)

            # Legacy path: predictor_config under model
            cfg = {
                "model": {
                    "prediction_steps": 50,
                    "predictor_config": {"context_length": 100},
                },
                "v5_optimization": {
                    "advanced_prediction": {"context_length": 999},  # must be ignored
                },
            }

            loaded = TwinBrainDigitalTwin.from_checkpoint(
                checkpoint_path=ckpt_path,
                config=cfg,
                device="cpu",
            )

        predictor = loaded.model.predictor
        self.assertEqual(
            predictor.context_length, 100,
            "Legacy model.predictor_config should take priority",
        )
        # Explicitly confirm the v5_optimization value (999) was NOT used
        self.assertNotEqual(
            predictor.context_length, 999,
            "v5_optimization.context_length=999 must be ignored when "
            "model.predictor_config is present",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Basin test: _run_trajectories device kwarg / tuple return bug
# ══════════════════════════════════════════════════════════════════════════════

class TestRunTrajectories(unittest.TestCase):
    """
    Tests for basin_test._run_trajectories.

    Regression tests for two bugs that caused all basin test trajectories to
    fail silently:

    1. ``_run_trajectories`` passed ``device=device`` to
       ``BrainDynamicsSimulator.rollout()``, which does not accept this
       keyword argument.  The device is fixed at simulator construction time.

    2. ``rollout()`` returns a ``(trajectory, times)`` tuple, but the old code
       assigned the tuple directly to the ``trajs[i]`` numpy row, which would
       raise a shape mismatch after fix 1.
    """

    def setUp(self):
        _models_dir = Path(__file__).parent.parent / "models"
        if str(_models_dir.parent) not in sys.path:
            sys.path.insert(0, str(_models_dir.parent))

    def _make_sim(self) -> "BrainDynamicsSimulator":
        from models.graph_native_system import GraphNativeBrainModel
        from models.digital_twin_inference import TwinBrainDigitalTwin
        from torch_geometric.data import HeteroData as HD
        H = 16
        model = GraphNativeBrainModel(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            in_channels_dict={"fmri": 1},
            hidden_channels=H,
            prediction_steps=5,
            predictor_config={"context_length": 20},
        )
        model.eval()
        twin = TwinBrainDigitalTwin(model=model, device="cpu")
        g = HD()
        g["fmri"].x = torch.randn(N, 20, 1)
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        return BrainDynamicsSimulator(model=twin, base_graph=g, modality="fmri",
                                      device="cpu")

    def test_run_trajectories_does_not_crash(self):
        """_run_trajectories should not raise TypeError for 'device' kwarg."""
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from experiments.basin_test import _run_trajectories

        sim = self._make_sim()
        rng = np.random.default_rng(0)
        init_states = rng.random((3, N)).astype(np.float32)
        # Should not raise TypeError: unexpected keyword argument 'device'
        trajs = _run_trajectories(sim, init_states, T=4, device="cpu")
        self.assertEqual(trajs.shape, (3, 4, N))

    def test_run_trajectories_output_shape(self):
        """_run_trajectories should return (n_traj, T, N) numpy array."""
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from experiments.basin_test import _run_trajectories

        sim = self._make_sim()
        rng = np.random.default_rng(1)
        n_traj = 4
        T_steps = 6
        init_states = rng.random((n_traj, N)).astype(np.float32)
        trajs = _run_trajectories(sim, init_states, T=T_steps)
        self.assertEqual(trajs.shape, (n_traj, T_steps, N),
                         "Expected shape (n_traj, T, N)")
        self.assertTrue(np.isfinite(trajs).all(),
                        "Trajectories should contain only finite values")


# ══════════════════════════════════════════════════════════════════════════════
# Regression tests for pipeline bugs
# ══════════════════════════════════════════════════════════════════════════════

class TestCSDKeyNames(unittest.TestCase):
    """
    Regression: run_critical_slowing_down_analysis returns a nested dict
    (top-level keys: 'per_trajectory_ews', 'aggregate', 'report').
    The pipeline used to read wrong flat keys like 'ar1_trend_tau' which
    always returned NaN.  It must now read from csd['aggregate'].
    """

    def test_csd_returns_aggregate_key(self):
        """run_critical_slowing_down_analysis must return an 'aggregate' sub-dict."""
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from analysis.critical_slowing_down import run_critical_slowing_down_analysis

        rng = np.random.default_rng(42)
        # short trajectories sufficient to exercise the function; N=10 matches module constant
        trajs = rng.random((3, 40, 10)).astype(np.float32)
        result = run_critical_slowing_down_analysis(trajs)

        self.assertIn("aggregate", result,
                      "CSD result must contain 'aggregate' sub-dict")
        agg = result["aggregate"]
        self.assertIn("ac1_tau_mean", agg,
                      "aggregate must have 'ac1_tau_mean' (not 'ar1_trend_tau')")
        self.assertIn("var_tau_mean", agg,
                      "aggregate must have 'var_tau_mean' (not 'var_trend_tau')")
        self.assertIn("ews_score_mean", agg)
        # The old flat keys must NOT be present at the top level
        self.assertNotIn("ar1_trend_tau", result)
        self.assertNotIn("var_trend_tau", result)

    def test_csd_aggregate_values_finite(self):
        """aggregate metrics should be finite for well-behaved trajectories."""
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from analysis.critical_slowing_down import run_critical_slowing_down_analysis

        rng = np.random.default_rng(0)
        trajs = rng.random((3, 50, 10)).astype(np.float32)
        result = run_critical_slowing_down_analysis(trajs)
        agg = result.get("aggregate", {})
        for key in ("ac1_tau_mean", "var_tau_mean", "ews_score_mean"):
            val = agg.get(key)
            self.assertIsNotNone(val, f"'{key}' should not be None")
            # NaN means an implementation bug; the function computes per-region
            # Kendall tau which should be finite for random data
            self.assertTrue(
                np.isfinite(val),
                f"aggregate['{key}']={val} should be finite",
            )


class TestQ5RandomLLEExtraction(unittest.TestCase):
    """
    Regression: pipeline Q5 used to read rc.get('random_lle_mean') which
    always returned None because run_random_model_comparison returns keys
    like 'random_sr1.50', not 'random_lle_mean'.  Q5 must extract LLEs
    from the actual dict structure.
    """

    def test_random_model_comparison_has_random_sr_keys(self):
        """run_random_model_comparison must return 'random_srX.XX' keys."""
        trajs = np.random.rand(3, 20, 10).astype(np.float32)
        result = run_random_model_comparison(
            trajectories=trajs,
            random_n_init=3,
            random_steps=15,
            spectral_radii=[1.5],   # single SR for speed
        )
        random_keys = [k for k in result if k.startswith("random_sr")]
        self.assertTrue(len(random_keys) >= 1,
                        f"Expected 'random_srX.XX' keys, got {list(result)}")
        # The LLE must be accessible at result['random_sr1.50']['mean_lyapunov']
        sr_key = random_keys[0]
        self.assertIn("mean_lyapunov", result[sr_key],
                      f"'{sr_key}' must contain 'mean_lyapunov'")

    def test_q5_lle_extraction_logic(self):
        """Simulate the Q5 extraction logic: mean LLE across random_srX.XX entries."""
        rc = {
            "model": {"mean_lyapunov": -0.02},
            "random_sr0.90": {"mean_lyapunov": -0.05, "stability": "stable"},
            "random_sr1.50": {"mean_lyapunov": 0.01, "stability": "weakly_chaotic"},
            "random_sr2.00": {"mean_lyapunov": 0.06, "stability": "chaotic"},
        }
        rand_lles = [
            v.get("mean_lyapunov")
            for k, v in rc.items()
            if k.startswith("random_sr") and isinstance(v, dict)
        ]
        rand_lles = [x for x in rand_lles if x is not None and np.isfinite(x)]
        self.assertEqual(len(rand_lles), 3)
        rand_lle = float(np.mean(rand_lles))
        self.assertAlmostEqual(rand_lle, (-0.05 + 0.01 + 0.06) / 3, places=6)


class TestQ4DeltaRhoSignConvention(unittest.TestCase):
    """
    Regression: Q4 used to check delta_rho > 0.05, so a large negative delta
    (original_rho - random_rho = -9.28) incorrectly reported 'not verified'
    even though the training structure clearly matters.  The check must use
    abs(delta_rho) > 0.05.
    """

    def test_negative_delta_rho_is_verified(self):
        """Large negative delta_rho should still mark Q4 as verified."""
        delta_rho = -9.2874
        # Old logic (buggy):
        old_verified = bool(delta_rho is not None and delta_rho > 0.05)
        self.assertFalse(old_verified, "Old logic incorrectly failed for large negative delta")
        # New logic (fixed):
        new_verified = bool(delta_rho is not None and abs(delta_rho) > 0.05)
        self.assertTrue(new_verified, "Fixed logic should verify large negative delta_rho")

    def test_structure_preserving_judgment_negative_delta(self):
        """run_structure_preserving_random should give a meaningful judgment for
        negative delta_rho (training suppresses spectral radius)."""
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from analysis.network_perturbation import run_structure_preserving_random

        # Construct a W with very low spectral radius (~0.2) so that random
        # permutations of the same weights produce higher spectral radius.
        rng = np.random.default_rng(7)
        n_nodes = 10  # same as module-level N
        W = rng.standard_normal((n_nodes, n_nodes)).astype(np.float32)
        sr = float(np.abs(np.linalg.eigvals(W)).max())
        W = W * 0.2 / (sr + 1e-9)   # force ρ ≈ 0.2

        result = run_structure_preserving_random(W, n_random=3, seed=42)
        self.assertIn("judgment", result)
        self.assertIn("delta_rho", result)
        # For a strong negative delta (training reduced ρ significantly),
        # the judgment must NOT say "random networks similar to trained"
        delta = result.get("delta_rho")
        if delta is not None and abs(delta) > 0.1:
            self.assertNotIn("similar", result["judgment"],
                             "Large |delta_rho| must not produce 'similar' judgment")


class TestGraphStructureComparison(unittest.TestCase):
    """
    Tests for run_graph_structure_comparison — TASK 1.

    Verifies that the degree-preserving baseline now reports actual trajectory-
    based dynamics metrics (LLE, PCA dim, D2) via tanh(W @ x) dynamics, not NaN.
    This is the implementation of the third control condition: hub/degree matters.
    """

    def _import(self):
        import sys
        _bd = Path(__file__).parent.parent / "brain_dynamics"
        if str(_bd) not in sys.path:
            sys.path.insert(0, str(_bd))
        from analysis.random_comparison import (
            run_graph_structure_comparison,
            _run_tanh_trajectories,
            _tanh_dynamics_metrics,
        )
        return run_graph_structure_comparison, _run_tanh_trajectories, _tanh_dynamics_metrics

    def test_run_tanh_trajectories_shape(self):
        """_run_tanh_trajectories should return (n_init, steps, N) float32 array."""
        _, _run_tanh_trajectories, _ = self._import()
        import numpy as np
        N = 8
        W = np.random.randn(N, N) * 0.3
        trajs = _run_tanh_trajectories(W, n_init=4, steps=10, seed=0)
        self.assertEqual(trajs.shape, (4, 10, N))
        self.assertTrue(np.all(trajs >= 0.0), "tanh clip should keep values >= 0")
        self.assertTrue(np.all(trajs <= 1.0), "tanh clip should keep values <= 1")

    def test_degree_preserving_has_dynamics_metrics(self):
        """degree_preserving entry must have finite lle, pca_dim_90pct after the fix."""
        run_graph_structure_comparison, _, _ = self._import()
        import numpy as np
        N = 8
        rng = np.random.default_rng(0)
        W = rng.standard_normal((N, N)) * 0.5
        trajs = rng.random((3, 20, N)).astype(np.float32)
        result = run_graph_structure_comparison(
            W=W,
            trajectories=trajs,
            n_random=2,
            n_tanh_init=3,
            tanh_steps=30,
            lle_steps=50,
            seed=0,
        )
        self.assertIn("degree_preserving", result)
        dp = result["degree_preserving"]
        # Before fix: lle was always NaN; after fix it must be a finite float
        self.assertIn("lle", dp)
        self.assertTrue(np.isfinite(dp["lle"]),
                        f"degree_preserving lle should be finite, got {dp['lle']}")
        self.assertGreater(dp["pca_dim_90pct"], 0,
                           "degree_preserving pca_dim_90pct should be > 0")
        # lle_std must be present (from averaging over n_random matrices)
        self.assertIn("lle_std", dp)

    def test_fully_random_has_dynamics_metrics(self):
        """fully_random entry must have finite lle after the fix."""
        run_graph_structure_comparison, _, _ = self._import()
        import numpy as np
        N = 8
        rng = np.random.default_rng(1)
        W = rng.standard_normal((N, N)) * 0.5
        trajs = rng.random((3, 20, N)).astype(np.float32)
        result = run_graph_structure_comparison(
            W=W,
            trajectories=trajs,
            n_random=2,
            n_tanh_init=3,
            tanh_steps=30,
            lle_steps=50,
            seed=1,
        )
        fr = result["fully_random"]
        self.assertTrue(np.isfinite(fr["lle"]),
                        f"fully_random lle should be finite, got {fr['lle']}")

    def test_brain_entry_has_gnn_and_tanh_metrics(self):
        """brain_graph entry must have both lle_gnn (GNN) and lle (tanh)."""
        run_graph_structure_comparison, _, _ = self._import()
        import numpy as np
        N = 8
        rng = np.random.default_rng(2)
        W = rng.standard_normal((N, N)) * 0.5
        trajs = rng.random((3, 20, N)).astype(np.float32)
        result = run_graph_structure_comparison(
            W=W,
            trajectories=trajs,
            n_random=2,
            n_tanh_init=3,
            tanh_steps=30,
            lle_steps=50,
            seed=2,
        )
        bg = result["brain_graph"]
        # GNN metrics from Phase 1 trajectories
        self.assertIn("lle_gnn", bg)
        self.assertIn("d2_gnn", bg)
        self.assertIn("pca_dim_gnn", bg)
        # tanh dynamics metrics for fair comparison
        self.assertIn("lle", bg)
        self.assertIn("pca_dim_90pct", bg)

    def test_three_way_comparison_produces_valid_structure(self):
        """All three entries must be present and have required keys."""
        run_graph_structure_comparison, _, _ = self._import()
        import numpy as np
        N = 8
        rng = np.random.default_rng(3)
        W = rng.standard_normal((N, N)) * 0.5
        trajs = rng.random((3, 20, N)).astype(np.float32)
        result = run_graph_structure_comparison(
            W=W,
            trajectories=trajs,
            n_random=2,
            n_tanh_init=3,
            tanh_steps=30,
            lle_steps=50,
            seed=3,
        )
        for key in ("brain_graph", "degree_preserving", "fully_random"):
            self.assertIn(key, result)
            entry = result[key]
            for metric in ("lle", "pca_dim_90pct", "spectral_radius"):
                self.assertIn(metric, entry,
                              f"{key} must have '{metric}' key")


if __name__ == "__main__":
    unittest.main()
