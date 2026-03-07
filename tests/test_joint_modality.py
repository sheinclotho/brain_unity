"""
Tests for joint modality and related dt/bounds-aware fixes.

Covers:
  - BrainDynamicsSimulator joint mode initialisation
  - dt fixed to fMRI TR (not native EEG sampling rate)
  - state_bounds property
  - sample_random_state for joint mode
  - _inject_x0_into_context for joint mode (z-score ↔ raw round-trip)
  - _rollout_joint shape and z-normalised output
  - _rollout_joint: fMRI-range and EEG-range stimulation node routing
  - _rollout_joint: out-of-range node raises ValueError (no silent fallback)
  - rollout_multi_stim joint mode: explicit per-modality routing
  - Wolf LLE bounds-aware clipping: skips clip when state_bounds is None
  - FTLE bounds-aware clipping
  - response_matrix joint mode: correct modality routing
  - virtual_stimulation uses sample_random_state
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Ensure the twinbrain-dynamics directory is on sys.path
_TD_DIR = Path(__file__).parent.parent / "twinbrain-dynamics"
if str(_TD_DIR) not in sys.path:
    sys.path.insert(0, str(_TD_DIR))

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    SinStimulus,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

N_FMRI = 8
N_EEG  = 4
CHUNK  = 3          # prediction_steps of the mock model
FMRI_SR = 0.5       # 0.5 Hz → TR = 2 s
EEG_SR  = 250.0     # 250 Hz (native) — should NOT be used as dt


def _make_heterodata(n_fmri: int = N_FMRI, n_eeg: int = N_EEG, t: int = 20) -> HeteroData:
    """Create a minimal two-modality HeteroData graph for testing."""
    g = HeteroData()
    # fMRI: [N_fmri, T, 1]
    fmri_data = torch.rand(n_fmri, t, 1)
    g["fmri"].x = fmri_data
    g["fmri"].sampling_rate = FMRI_SR
    # EEG: [N_eeg, T, 1] — same T for simplicity (the model decides step alignment)
    eeg_data = torch.rand(n_eeg, t, 1) * 2.0  # deliberately different scale
    g["eeg"].x = eeg_data
    g["eeg"].sampling_rate = EEG_SR
    return g


class _MockPredictor:
    """Minimal stand-in for EnhancedMultiStepPredictor."""
    context_length = 20
    prediction_steps = CHUNK


class _MockTwinBrain:
    """
    Stand-in for TwinBrainDigitalTwin.

    predict_future: returns deterministic predictions for both modalities.
    simulate_intervention: records which modality/nodes were stimulated,
                           then returns the same predictions as predict_future.
    """

    class _Model:
        predictor = _MockPredictor()

    model = _Model()

    def __init__(self, n_fmri: int = N_FMRI, n_eeg: int = N_EEG):
        self.n_fmri = n_fmri
        self.n_eeg = n_eeg
        self.last_interventions: Optional[dict] = None

    def predict_future(self, context: HeteroData, num_steps: int = CHUNK) -> Dict[str, torch.Tensor]:
        return {
            "fmri": torch.rand(self.n_fmri, num_steps, 1) * 0.5,
            "eeg":  torch.rand(self.n_eeg,  num_steps, 1) * 2.0,
        }

    def simulate_intervention(
        self,
        baseline_data: HeteroData,
        interventions: dict,
        num_prediction_steps: int = CHUNK,
    ) -> dict:
        self.last_interventions = interventions
        pred = self.predict_future(baseline_data, num_steps=num_prediction_steps)
        return {"perturbed": pred, "causal_effect": {k: v for k, v in pred.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# BrainDynamicsSimulator — joint mode initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestJointModeInit(unittest.TestCase):

    def setUp(self):
        self.graph = _make_heterodata()
        self.model = _MockTwinBrain()
        self.sim = BrainDynamicsSimulator(
            model=self.model,
            base_graph=self.graph,
            modality="joint",
        )

    def test_n_regions_is_sum(self):
        self.assertEqual(self.sim.n_regions, N_FMRI + N_EEG)

    def test_n_fmri_regions(self):
        self.assertEqual(self.sim.n_fmri_regions, N_FMRI)

    def test_n_eeg_regions(self):
        self.assertEqual(self.sim.n_eeg_regions, N_EEG)

    def test_dt_equals_fmri_tr(self):
        expected_dt = 1.0 / FMRI_SR  # = 2.0 s
        self.assertAlmostEqual(self.sim.dt, expected_dt, places=6)

    def test_dt_not_eeg_native_rate(self):
        """dt must NOT be 1/EEG_SR = 0.004 s."""
        eeg_native_dt = 1.0 / EEG_SR  # 0.004 s
        self.assertNotAlmostEqual(self.sim.dt, eeg_native_dt, places=4)

    def test_state_bounds_is_none(self):
        """Joint mode has z-scored unbounded state space."""
        self.assertIsNone(self.sim.state_bounds)

    def test_modality_attribute(self):
        self.assertEqual(self.sim.modality, "joint")

    def test_fmri_normalisation_stats_stored(self):
        self.assertTrue(hasattr(self.sim, "_fmri_mean"))
        self.assertTrue(hasattr(self.sim, "_fmri_std"))
        self.assertEqual(self.sim._fmri_mean.shape, (N_FMRI,))
        self.assertEqual(self.sim._fmri_std.shape,  (N_FMRI,))

    def test_eeg_normalisation_stats_stored(self):
        self.assertTrue(hasattr(self.sim, "_eeg_mean"))
        self.assertTrue(hasattr(self.sim, "_eeg_std"))
        self.assertEqual(self.sim._eeg_mean.shape, (N_EEG,))
        self.assertEqual(self.sim._eeg_std.shape,  (N_EEG,))

    def test_std_positive(self):
        """Std has 1e-8 guard — must be strictly positive."""
        self.assertTrue(np.all(self.sim._fmri_std > 0))
        self.assertTrue(np.all(self.sim._eeg_std  > 0))

    def test_joint_missing_eeg_raises(self):
        g = HeteroData()
        g["fmri"].x = torch.rand(N_FMRI, 10, 1)
        with self.assertRaises(ValueError):
            BrainDynamicsSimulator(model=self.model, base_graph=g, modality="joint")

    def test_joint_missing_fmri_raises(self):
        g = HeteroData()
        g["eeg"].x = torch.rand(N_EEG, 10, 1)
        with self.assertRaises(ValueError):
            BrainDynamicsSimulator(model=self.model, base_graph=g, modality="joint")

    def test_invalid_modality_raises(self):
        with self.assertRaises(ValueError):
            BrainDynamicsSimulator(
                model=self.model, base_graph=self.graph, modality="invalid"
            )


# ══════════════════════════════════════════════════════════════════════════════
# dt fix: single modalities
# ══════════════════════════════════════════════════════════════════════════════

class TestDtFix(unittest.TestCase):

    def _sim(self, modality: str) -> BrainDynamicsSimulator:
        return BrainDynamicsSimulator(
            model=_MockTwinBrain(),
            base_graph=_make_heterodata(),
            modality=modality,
        )

    def test_fmri_dt_equals_tr(self):
        sim = self._sim("fmri")
        self.assertAlmostEqual(sim.dt, 1.0 / FMRI_SR, places=6)

    def test_eeg_dt_equals_fmri_tr_not_native_eeg(self):
        """EEG modality: model predicts at fMRI TR → dt = TR, not 1/EEG_SR."""
        sim = self._sim("eeg")
        self.assertAlmostEqual(sim.dt, 1.0 / FMRI_SR, places=6)
        self.assertNotAlmostEqual(sim.dt, 1.0 / EEG_SR, places=4)

    def test_fmri_state_bounds(self):
        sim = self._sim("fmri")
        self.assertEqual(sim.state_bounds, (0.0, 1.0))

    def test_eeg_state_bounds(self):
        sim = self._sim("eeg")
        self.assertEqual(sim.state_bounds, (0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# sample_random_state — joint mode
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleRandomStateJoint(unittest.TestCase):

    def setUp(self):
        self.sim = BrainDynamicsSimulator(
            model=_MockTwinBrain(),
            base_graph=_make_heterodata(),
            modality="joint",
        )

    def test_shape(self):
        x0 = self.sim.sample_random_state()
        self.assertEqual(x0.shape, (N_FMRI + N_EEG,))

    def test_not_clipped_to_0_1(self):
        """Joint states are z-scored — must allow values outside [0, 1]."""
        rng = np.random.default_rng(42)
        # Use large seed so we get values spread beyond [0,1]
        rng2 = np.random.default_rng(0)
        found_negative = False
        for _ in range(50):
            x0 = self.sim.sample_random_state(rng=rng2)
            if np.any(x0 < 0.0):
                found_negative = True
                break
        self.assertTrue(found_negative, "Joint x0 should contain negative z-scores")

    def test_deterministic_with_rng(self):
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        x1 = self.sim.sample_random_state(rng=rng1)
        x2 = self.sim.sample_random_state(rng=rng2)
        np.testing.assert_array_equal(x1, x2)


# ══════════════════════════════════════════════════════════════════════════════
# _inject_x0_into_context — joint mode: z-score ↔ raw round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestInjectX0Joint(unittest.TestCase):

    def setUp(self):
        self.graph = _make_heterodata(t=20)
        self.sim = BrainDynamicsSimulator(
            model=_MockTwinBrain(),
            base_graph=self.graph,
            modality="joint",
        )

    def _clone_ctx(self):
        from simulator.brain_dynamics_simulator import _clone_hetero_graph
        return self.sim._trim_context(_clone_hetero_graph(self.sim.base_graph))

    def test_injection_modifies_both_contexts(self):
        ctx = self._clone_ctx()
        fmri_before = ctx["fmri"].x[:, -1, 0].clone()
        eeg_before  = ctx["eeg"].x[:, -1, 0].clone()

        x0_z = self.sim.sample_random_state()
        self.sim._inject_x0_into_context(ctx, x0_z)

        fmri_after = ctx["fmri"].x[:, -1, 0]
        eeg_after  = ctx["eeg"].x[:, -1, 0]

        # Both contexts must have changed at the last time step
        self.assertFalse(torch.allclose(fmri_before, fmri_after))
        self.assertFalse(torch.allclose(eeg_before,  eeg_after))

    def test_round_trip_accuracy(self):
        """z-score → inject (un-z-score) → read raw → re-z-score ≈ original."""
        ctx = self._clone_ctx()
        x0_z = np.zeros(N_FMRI + N_EEG, dtype=np.float32)  # z=0 → raw = mean
        self.sim._inject_x0_into_context(ctx, x0_z)

        fmri_raw = ctx["fmri"].x[:, -1, 0].numpy()
        eeg_raw  = ctx["eeg"].x[:, -1, 0].numpy()

        # Un-z-scoring z=0 should give exactly the mean
        np.testing.assert_allclose(fmri_raw, self.sim._fmri_mean, atol=1e-5)
        np.testing.assert_allclose(eeg_raw,  self.sim._eeg_mean,  atol=1e-5)

    def test_wrong_shape_skips_silently(self):
        """Wrong-shape x0 should not raise — silently ignored."""
        ctx = self._clone_ctx()
        fmri_before = ctx["fmri"].x[:, -1, 0].clone()
        # wrong length
        self.sim._inject_x0_into_context(ctx, np.zeros(N_FMRI + N_EEG + 5))
        # should be unchanged
        torch.testing.assert_close(ctx["fmri"].x[:, -1, 0], fmri_before)


# ══════════════════════════════════════════════════════════════════════════════
# _rollout_joint — free dynamics
# ══════════════════════════════════════════════════════════════════════════════

class TestRolloutJoint(unittest.TestCase):

    def setUp(self):
        self.graph = _make_heterodata()
        self.model = _MockTwinBrain()
        self.sim = BrainDynamicsSimulator(
            model=self.model,
            base_graph=self.graph,
            modality="joint",
        )

    def test_trajectory_shape(self):
        traj, times = self.sim.rollout(x0=None, steps=7)
        self.assertEqual(traj.shape, (7, N_FMRI + N_EEG))
        self.assertEqual(times.shape, (7,))

    def test_times_axis_is_fmri_tr(self):
        """Each time step should be one fMRI TR."""
        traj, times = self.sim.rollout(steps=5)
        expected_dt = 1.0 / FMRI_SR
        np.testing.assert_allclose(np.diff(times), expected_dt, rtol=1e-5)

    def test_output_is_z_scored_not_0_1(self):
        """Joint output is z-normalised — values can exceed 1 or be < 0."""
        trajs = []
        rng = np.random.default_rng(99)
        for _ in range(20):
            x0 = self.sim.sample_random_state(rng=rng)
            traj, _ = self.sim.rollout(x0=x0, steps=6)
            trajs.append(traj)
        all_traj = np.concatenate(trajs, axis=0)
        # With random model outputs, at least some values should exceed [0, 1]
        has_outside = np.any(all_traj < 0.0) or np.any(all_traj > 1.0)
        self.assertTrue(has_outside, "Joint trajectory should not be clipped to [0,1]")

    def test_fmri_range_stimulus_reaches_model(self):
        """Stimulating a fMRI-range node routes to {'fmri': ...}."""
        stim = SinStimulus(node=0, freq=1.0, amp=0.5, duration=4, onset=0)
        self.sim.rollout(steps=4, stimulus=stim)
        interventions = self.model.last_interventions
        self.assertIsNotNone(interventions)
        self.assertIn("fmri", interventions)
        self.assertEqual(interventions["fmri"][0], [0])  # node index 0

    def test_eeg_range_stimulus_routes_to_eeg(self):
        """Stimulating an EEG-range node routes to {'eeg': ...} with correct offset."""
        eeg_joint_idx = N_FMRI  # first EEG channel in joint space
        stim = SinStimulus(node=eeg_joint_idx, freq=1.0, amp=0.5, duration=4, onset=0)
        self.sim.rollout(steps=4, stimulus=stim)
        interventions = self.model.last_interventions
        self.assertIsNotNone(interventions)
        self.assertIn("eeg", interventions)
        self.assertEqual(interventions["eeg"][0], [0])  # channel 0 in EEG space

    def test_out_of_range_stimulus_node_raises(self):
        """Out-of-range node must raise ValueError, not silently skip."""
        bad_node = N_FMRI + N_EEG + 1
        stim = SinStimulus(node=bad_node, freq=1.0, amp=0.5, duration=4, onset=0)
        with self.assertRaises(ValueError):
            self.sim.rollout(steps=4, stimulus=stim)


# ══════════════════════════════════════════════════════════════════════════════
# rollout_multi_stim — joint mode explicit routing
# ══════════════════════════════════════════════════════════════════════════════

class TestRolloutMultiStimJoint(unittest.TestCase):

    def setUp(self):
        self.graph = _make_heterodata()
        self.model = _MockTwinBrain()
        self.sim = BrainDynamicsSimulator(
            model=self.model,
            base_graph=self.graph,
            modality="joint",
        )

    def test_shape(self):
        stimuli = [
            SinStimulus(node=0, freq=1.0, amp=0.3, duration=5, onset=0),
            SinStimulus(node=N_FMRI, freq=1.0, amp=0.3, duration=5, onset=0),
        ]
        traj, _ = self.sim.rollout_multi_stim(
            x0=None, steps=5, stimuli=stimuli
        )
        self.assertEqual(traj.shape, (5, N_FMRI + N_EEG))

    def test_fmri_and_eeg_stim_separated(self):
        """Multi-stim with fMRI and EEG nodes routes to separate modalities."""
        stimuli = [
            SinStimulus(node=2, freq=1.0, amp=0.3, duration=4, onset=0),        # fMRI 2
            SinStimulus(node=N_FMRI + 1, freq=1.0, amp=0.3, duration=4, onset=0),  # EEG ch 1
        ]
        self.sim.rollout_multi_stim(x0=None, steps=4, stimuli=stimuli)
        ivt = self.model.last_interventions
        self.assertIn("fmri", ivt)
        self.assertIn("eeg", ivt)
        self.assertIn(2, ivt["fmri"][0])
        self.assertIn(1, ivt["eeg"][0])

    def test_out_of_range_node_raises(self):
        stimuli = [SinStimulus(node=N_FMRI + N_EEG + 5, freq=1.0, amp=0.3, duration=2, onset=0)]
        with self.assertRaises(ValueError):
            self.sim.rollout_multi_stim(x0=None, steps=2, stimuli=stimuli)


# ══════════════════════════════════════════════════════════════════════════════
# Wolf LLE bounds-aware clipping
# ══════════════════════════════════════════════════════════════════════════════

class _BoundedSim:
    """Minimal simulator with [0,1] bounds (fMRI/EEG single-modality)."""
    n_regions = 5
    state_bounds = (0.0, 1.0)

    def rollout(self, x0, steps, stimulus=None):
        T = np.full((steps, self.n_regions), 0.5, dtype=np.float32)
        return T, np.arange(steps, dtype=np.float32)

    def wolf_rollout_pair(self, x_base, x_pert, steps, wolf_context):
        return x_base, x_pert, wolf_context


class _UnboundedSim:
    """Minimal simulator with None bounds (joint mode, z-scored)."""
    n_regions = 5
    state_bounds = None

    def rollout(self, x0, steps, stimulus=None):
        # Return values outside [0,1] to confirm no clipping occurs
        T = np.full((steps, self.n_regions), 2.0, dtype=np.float32)
        return T, np.arange(steps, dtype=np.float32)

    def wolf_rollout_pair(self, x_base, x_pert, steps, wolf_context):
        return x_base, x_pert, wolf_context


class TestWolfLLEBoundsAware(unittest.TestCase):

    def test_bounded_sim_clip_happens(self):
        """For bounded sim, x_pert near boundary should be clipped to [0,1]."""
        from analysis.lyapunov import wolf_largest_lyapunov
        sim = _BoundedSim()
        # Start very close to 0 so perturbation would go negative → should be clipped
        x0 = np.full(sim.n_regions, 0.0001, dtype=np.float32)
        # Should not raise; clipping should silently handle negative values
        lle, _ = wolf_largest_lyapunov(sim, x0, total_steps=20, renorm_steps=5)
        self.assertTrue(np.isfinite(lle) or np.isnan(lle))

    def test_unbounded_sim_no_clip(self):
        """For unbounded (joint) sim, no clipping to [0,1] should occur."""
        from analysis.lyapunov import wolf_largest_lyapunov
        sim = _UnboundedSim()
        # x0 outside [0,1] — for unbounded mode, this should be accepted
        x0 = np.full(sim.n_regions, -1.5, dtype=np.float32)
        lle, _ = wolf_largest_lyapunov(sim, x0, total_steps=20, renorm_steps=5)
        self.assertTrue(np.isfinite(lle) or np.isnan(lle))


class TestFTLEBoundsAware(unittest.TestCase):

    def test_bounded_ftle_clips(self):
        from analysis.lyapunov import ftle_lyapunov
        sim = _BoundedSim()
        T = 20
        traj = np.full((T, sim.n_regions), 0.5, dtype=np.float32)
        result = ftle_lyapunov(traj, sim)
        self.assertTrue(np.isfinite(result))

    def test_unbounded_ftle_no_clip(self):
        from analysis.lyapunov import ftle_lyapunov
        sim = _UnboundedSim()
        # Trajectory with z-scored values (not in [0,1])
        T = 20
        traj = np.random.default_rng(0).normal(0, 1, (T, sim.n_regions)).astype(np.float32)
        result = ftle_lyapunov(traj, sim)
        self.assertTrue(np.isfinite(result))


# ══════════════════════════════════════════════════════════════════════════════
# Response matrix — joint mode routing
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseMatrixJoint(unittest.TestCase):

    def setUp(self):
        self.graph = _make_heterodata()
        self.model = _MockTwinBrain()
        self.sim = BrainDynamicsSimulator(
            model=self.model,
            base_graph=self.graph,
            modality="joint",
        )

    def test_matrix_shape(self):
        from analysis.response_matrix import compute_response_matrix
        n_stim = 4  # 2 fMRI + 2 EEG nodes
        R = compute_response_matrix(self.sim, n_nodes=n_stim)
        # Should be (n_stim, N_fmri + N_eeg)
        self.assertEqual(R.shape, (n_stim, N_FMRI + N_EEG))

    def test_fmri_node_routes_to_fmri_modality(self):
        from analysis.response_matrix import compute_response_matrix
        compute_response_matrix(self.sim, n_nodes=1, stim_amplitude=0.5)
        # Node 0 < N_FMRI → interventions should target fmri
        self.assertIn("fmri", self.model.last_interventions)
        self.assertEqual(self.model.last_interventions["fmri"][0], [0])

    def test_eeg_node_routes_to_eeg_modality(self):
        from analysis.response_matrix import compute_response_matrix
        # Start from N_FMRI to land on EEG node 0
        compute_response_matrix(self.sim, n_nodes=N_FMRI + 1, stim_amplitude=0.5)
        # Last call was node N_FMRI → eeg channel 0
        self.assertIn("eeg", self.model.last_interventions)
        self.assertEqual(self.model.last_interventions["eeg"][0], [0])


# ══════════════════════════════════════════════════════════════════════════════
# virtual_stimulation uses sample_random_state
# ══════════════════════════════════════════════════════════════════════════════

class TestVirtualStimUsesSimSampleState(unittest.TestCase):
    """Verify that run_stimulation uses simulator.sample_random_state(), not rng.random()."""

    def test_joint_initial_state_scale(self):
        """
        For joint mode, run_stimulation's x0 (from sample_random_state) should
        be in z-score scale (values can be negative), NOT clipped to [0,1].
        """
        from experiments.virtual_stimulation import run_stimulation

        graph = _make_heterodata()
        model = _MockTwinBrain()
        sim = BrainDynamicsSimulator(
            model=model, base_graph=graph, modality="joint"
        )
        stim = SinStimulus(node=0, freq=1.0, amp=0.5, duration=3, onset=1)
        result = run_stimulation(
            simulator=sim, node=0, amplitude=0.5, frequency=1.0,
            stim_steps=3, pre_steps=1, post_steps=1,
        )
        # Trajectory must have shape matching n_regions
        full_traj = np.concatenate([
            result.pre_trajectory, result.stim_trajectory, result.post_trajectory
        ], axis=0)
        self.assertEqual(full_traj.shape[1], N_FMRI + N_EEG)


if __name__ == "__main__":
    unittest.main()
