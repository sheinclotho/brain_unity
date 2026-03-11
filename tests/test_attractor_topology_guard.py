import sys
import unittest
from pathlib import Path


_BD_DIR = Path(__file__).parent.parent / "brain_dynamics"
if str(_BD_DIR) not in sys.path:
    sys.path.insert(0, str(_BD_DIR))

from analysis.attractor_topology import _score_hypotheses
from experiments.attractor_analysis import _uniform_cluster_check


class TestContinuousAttractorGuard(unittest.TestCase):
    def test_uniform_cluster_check_not_triggered_for_small_k(self):
        suspect, _ = _uniform_cluster_check(
            basin={0: 0.50, 1: 0.50},
            silhouette=0.97,
        )
        self.assertFalse(
            suspect,
            "K=2 near-uniform splits are common and must not be treated as "
            "ring/continuous-attractor evidence.",
        )

    def test_scoring_suppresses_single_clue_ca_false_positive(self):
        # Only one CA clue (neutral direction) should not dominate final result.
        scoring = _score_hypotheses(
            spectral={"freq_classification": "limit_cycle", "n_total_peaks": 2},
            velocity={
                "rotation_index": 0.65,
                "has_neutral_direction": True,   # single CA clue
                "rotation_score": 1.8,
            },
            rqa={"DET": 0.95, "LAM": 0.55},
            local_dim={"local_dim_mean": 2.0},
            rosenstein_lle=-0.03,
            n_pca_90=2,
            period_stability={
                "stability_class": "stable_limit_cycle",
                "period_cv": 0.03,
                "slow_envelope_variation": 0.08,
            },
            kmeans_uniform_suspect=False,
            dmd_n_hopf_pairs=1,
        )
        self.assertNotEqual(scoring["top_hypothesis"], "CA")
        self.assertIn(scoring["top_hypothesis"], {"LC", "SM", "FP", "QP", "SA"})


if __name__ == "__main__":
    unittest.main()
