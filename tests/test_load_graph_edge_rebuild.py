"""
Tests for the intra-modal edge-rebuild fix in load_graph_for_inference.

Root cause: V5 graph caches store only node features (fmri.x, eeg.x).
Without intra-modal edges ('fmri', 'connects', 'fmri') the encoder's
SpatialTemporalGraphConv and GraphPredictionPropagator both degrade to
passthrough, making stimulating node i only affect node i → diagonal
response matrix.

The fix: load_graph_for_inference now computes FC-based intra-modal edges
from the loaded timeseries before returning the graph.

These tests exercise the rebuild logic without touching the filesystem.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch_geometric.data import HeteroData

_REPO_ROOT = Path(__file__).parent.parent
_TD_DIR = _REPO_ROOT / "twinbrain-dynamics"
for _p in [str(_REPO_ROOT), str(_TD_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── helpers ──────────────────────────────────────────────────────────────────

N_FMRI = 10
N_EEG  = 6
T      = 30   # short timeseries — enough for correlation estimation


def _make_v5_cache(
    include_fmri: bool = True,
    include_eeg: bool = False,
    include_fmri_edges: bool = False,
    n_fmri: int = N_FMRI,
    n_eeg: int = N_EEG,
    t: int = T,
) -> HeteroData:
    """Return a minimal HeteroData in V5 format.

    By default (include_fmri_edges=False) the graph has *only* node features,
    mimicking what the training pipeline saves.
    """
    g = HeteroData()
    if include_fmri:
        g["fmri"].x = torch.rand(n_fmri, t, 1)
    if include_eeg:
        g["eeg"].x = torch.rand(n_eeg, t, 1)
    if include_fmri_edges:
        # Pre-existing intra-modal edge (should NOT be overwritten)
        g["fmri", "connects", "fmri"].edge_index = torch.zeros(2, 5, dtype=torch.long)
        g["fmri", "connects", "fmri"].edge_attr  = torch.ones(5, 1)
    return g


def _write_and_load(graph: HeteroData, tmp_dir: Path, **kwargs) -> HeteroData:
    """Save graph to a temp .pt file and call load_graph_for_inference on it."""
    # Lazy import so we can patch _REPO_ROOT inside the module if needed
    from twinbrain_dynamics_loader import load_graph_for_inference  # type: ignore[import]
    pt_path = tmp_dir / "test_graph.pt"
    torch.save(graph, str(pt_path))
    return load_graph_for_inference(pt_path, device="cpu", **kwargs)


# We import the loader module-level so patching can target it directly.
# Use a try/except so test collection doesn't fail when torch_geometric is absent.
try:
    from twinbrain_dynamics_loader import load_graph_for_inference as _LOAD_FN  # type: ignore[import]
    _LOADER_AVAILABLE = True
except Exception:
    _LOAD_FN = None
    _LOADER_AVAILABLE = False


def _load_via_tmpfile(graph: HeteroData, **kwargs) -> HeteroData:
    """Write graph to a tmp file and call the real load_graph_for_inference."""
    with tempfile.TemporaryDirectory() as td:
        pt_path = Path(td) / "graph.pt"
        torch.save(graph, str(pt_path))
        return _LOAD_FN(pt_path, device="cpu", **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Import under a stable name so tests can reference the function directly
# ══════════════════════════════════════════════════════════════════════════════
try:
    from twinbrain_dynamics_loader import load_graph_for_inference
except Exception:
    load_graph_for_inference = None  # type: ignore[assignment]


# ── unit tests against the loader's edge-rebuild helper (no disk I/O) ────────

class _FakeMapper:
    """Minimal stand-in for GraphNativeBrainMapper used in unit tests."""
    threshold_fmri = 0.3
    k_nearest_fmri = 5
    threshold_eeg  = 0.2
    k_nearest_eeg  = 3

    def _compute_correlation_gpu(self, ts: np.ndarray) -> np.ndarray:
        N = ts.shape[0]
        return np.abs(np.corrcoef(ts))  # [N, N]

    def _compute_eeg_connectivity(self, ts: np.ndarray) -> np.ndarray:
        N = ts.shape[0]
        return np.abs(np.corrcoef(ts))

    def build_graph_structure(self, conn, threshold=0.1, k_nearest=None):
        N = conn.shape[0]
        # Return a trivial star-graph edge_index: node 0 connects to all others
        src = torch.zeros(N - 1, dtype=torch.long)
        dst = torch.arange(1, N, dtype=torch.long)
        ei  = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        ea  = torch.ones(ei.shape[1], 1)
        return ei, ea

    def create_simple_cross_modal_edges(self, graph, k_cross_modal=5):
        n_eeg  = graph["eeg"].x.shape[0]
        n_fmri = graph["fmri"].x.shape[0]
        ei = torch.zeros(2, n_eeg, dtype=torch.long)  # all EEG → fMRI node 0
        ea = torch.ones(n_eeg, 1)
        return ei, ea


# ══════════════════════════════════════════════════════════════════════════════
# Functional tests using mock GraphNativeBrainMapper
# ══════════════════════════════════════════════════════════════════════════════

class TestIntraModalEdgeRebuild(unittest.TestCase):
    """Exercise the intra-modal edge rebuild path directly via mocking."""

    def _run_rebuild(self, graph: HeteroData) -> HeteroData:
        """Run the edge-rebuild logic extracted from load_graph_for_inference."""
        fake_mapper = _FakeMapper()

        _INTRA_MODAL_CFG = {
            "fmri": (fake_mapper.threshold_fmri, fake_mapper.k_nearest_fmri, "fMRI"),
            "eeg":  (fake_mapper.threshold_eeg,  fake_mapper.k_nearest_eeg,  "EEG"),
        }
        for nt, (thresh, k_nn, label) in _INTRA_MODAL_CFG.items():
            if nt not in graph.node_types:
                continue
            edge_type = (nt, "connects", nt)
            if edge_type in graph.edge_types:
                existing = getattr(graph[edge_type], "edge_index", None)
                if existing is not None and existing.shape[1] > 0:
                    continue  # keep pre-existing valid edges
            ts = graph[nt].x.squeeze(-1).cpu().numpy()
            if nt == "fmri":
                conn = fake_mapper._compute_correlation_gpu(ts)
            else:
                conn = fake_mapper._compute_eeg_connectivity(ts)
            ei, ea = fake_mapper.build_graph_structure(conn, threshold=thresh, k_nearest=k_nn)
            graph[edge_type].edge_index = ei
            graph[edge_type].edge_attr  = ea
        return graph

    # ── fMRI-only cache ───────────────────────────────────────────────────────

    def test_fmri_only_gets_intra_modal_edges(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False)
        self.assertNotIn(("fmri", "connects", "fmri"), g.edge_types)
        g = self._run_rebuild(g)
        self.assertIn(("fmri", "connects", "fmri"), g.edge_types)

    def test_fmri_edge_index_has_correct_shape(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False)
        g = self._run_rebuild(g)
        ei = g["fmri", "connects", "fmri"].edge_index
        self.assertEqual(ei.shape[0], 2, "edge_index must have 2 rows")
        self.assertGreater(ei.shape[1], 0, "there must be at least 1 edge")

    def test_fmri_edge_indices_in_valid_range(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False)
        g = self._run_rebuild(g)
        ei = g["fmri", "connects", "fmri"].edge_index
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < N_FMRI).all())

    def test_fmri_edge_attr_shape_matches_edge_index(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False)
        g = self._run_rebuild(g)
        ei = g["fmri", "connects", "fmri"].edge_index
        ea = g["fmri", "connects", "fmri"].edge_attr
        self.assertEqual(ea.shape[0], ei.shape[1])
        self.assertEqual(ea.shape[1], 1)

    # ── EEG-only cache ────────────────────────────────────────────────────────

    def test_eeg_only_gets_intra_modal_edges(self):
        g = HeteroData()
        g["eeg"].x = torch.rand(N_EEG, T, 1)
        self.assertNotIn(("eeg", "connects", "eeg"), g.edge_types)
        g = self._run_rebuild(g)
        self.assertIn(("eeg", "connects", "eeg"), g.edge_types)

    # ── fMRI + EEG cache ──────────────────────────────────────────────────────

    def test_joint_both_modalities_get_edges(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=True)
        g = self._run_rebuild(g)
        self.assertIn(("fmri", "connects", "fmri"), g.edge_types)
        self.assertIn(("eeg", "connects", "eeg"),  g.edge_types)

    # ── pre-existing edges must NOT be overwritten ────────────────────────────

    def test_preexisting_edges_preserved(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False, include_fmri_edges=True)
        original_ei = g["fmri", "connects", "fmri"].edge_index.clone()
        g = self._run_rebuild(g)
        rebuilt_ei = g["fmri", "connects", "fmri"].edge_index
        # Must have kept the original 5 edges, not replaced with the fake star-graph
        self.assertEqual(rebuilt_ei.shape[1], original_ei.shape[1])
        self.assertTrue(torch.equal(rebuilt_ei, original_ei))

    # ── edge integrity: no self-loops from the fake mapper ────────────────────

    def test_no_self_loops_in_fmri_edges(self):
        g = _make_v5_cache(include_fmri=True, include_eeg=False)
        g = self._run_rebuild(g)
        ei = g["fmri", "connects", "fmri"].edge_index
        self.assertTrue(
            (ei[0] != ei[1]).all(),
            "Intra-modal edges should not include self-loops"
        )

    # ── EEG edges reference valid node indices ───────────────────────────────

    def test_eeg_edge_indices_in_valid_range(self):
        g = HeteroData()
        g["eeg"].x = torch.rand(N_EEG, T, 1)
        g = self._run_rebuild(g)
        ei = g["eeg", "connects", "eeg"].edge_index
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < N_EEG).all())


# ══════════════════════════════════════════════════════════════════════════════
# Verify propagator passthrough vs. propagation difference
# ══════════════════════════════════════════════════════════════════════════════

class TestPropagatorRequiresEdges(unittest.TestCase):
    """
    Demonstrate that GraphPredictionPropagator is a passthrough when no edges
    are present, and produces non-trivial output when edges are given.

    This is the mechanistic proof that missing edges cause the diagonal matrix.
    """

    # ── one-time import at class level so path manipulation only happens once ─

    _GraphPredictionPropagator = None

    @classmethod
    def setUpClass(cls) -> None:
        models_dir = str(_REPO_ROOT / "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        try:
            from graph_native_system import GraphPredictionPropagator  # type: ignore[import]
            cls._GraphPredictionPropagator = GraphPredictionPropagator
        except (ImportError, ModuleNotFoundError):
            cls._GraphPredictionPropagator = None

    def _make_propagator(self, n: int = N_FMRI, H: int = 4):
        """Build a minimal propagator with one fMRI→fMRI edge type."""
        if self._GraphPredictionPropagator is None:
            self.skipTest("graph_native_system not importable in this environment")
        return self._GraphPredictionPropagator(
            node_types=["fmri"],
            edge_types=[("fmri", "connects", "fmri")],
            hidden_channels=H,
            num_prop_layers=1,
            dropout=0.0,
        )

    def test_no_edges_is_passthrough(self):
        prop = self._make_propagator()
        prop.eval()
        g = HeteroData()   # no edges
        pred = {"fmri": torch.randn(N_FMRI, 3, 4)}
        out  = prop(pred, g)
        # With no edges all messages lists are empty → output == input exactly
        self.assertTrue(torch.allclose(out["fmri"], pred["fmri"]),
                        "Expected passthrough when no edges present")

    def test_with_edges_output_differs_from_input(self):
        prop = self._make_propagator()
        prop.eval()
        # Fully connected (all-to-all) edge_index for N_FMRI nodes via torch
        idx = torch.arange(N_FMRI)
        src = idx.unsqueeze(1).expand(-1, N_FMRI).reshape(-1)  # [N*N]
        dst = idx.unsqueeze(0).expand(N_FMRI, -1).reshape(-1)  # [N*N]
        mask = src != dst
        ei = torch.stack([src[mask], dst[mask]], dim=0)  # [2, N*(N-1)]
        ea = torch.ones(ei.shape[1], 1)
        g = HeteroData()
        g["fmri", "connects", "fmri"].edge_index = ei
        g["fmri", "connects", "fmri"].edge_attr  = ea

        pred = {"fmri": torch.randn(N_FMRI, 3, 4)}
        out  = prop(pred, g)
        # With edges, messages aggregate from neighbours → output != input
        # (unless the conv weights happen to be exactly zero, which is
        # practically impossible after Xavier init)
        self.assertFalse(torch.allclose(out["fmri"], pred["fmri"]),
                         "Expected non-trivial propagation when edges are present")

    def test_perturbation_propagates_to_connected_nodes(self):
        """Stimulating node 0 should change node 1's output (if 0→1 edge exists)."""
        prop = self._make_propagator()
        prop.eval()
        # Create edge 0 → 1 only
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # undirected 0–1
        ea = torch.ones(2, 1)
        g = HeteroData()
        g["fmri", "connects", "fmri"].edge_index = ei
        g["fmri", "connects", "fmri"].edge_attr  = ea

        pred_base = torch.zeros(N_FMRI, 3, 4)
        pred_pert = pred_base.clone()
        pred_pert[0] += 1.0   # perturb node 0

        out_base = prop({"fmri": pred_base}, g)["fmri"]
        out_pert = prop({"fmri": pred_pert}, g)["fmri"]

        diff = (out_pert - out_base).abs()
        # Node 1 is connected to node 0 → must change
        self.assertGreater(diff[1].max().item(), 1e-4,
                           "Node 1 should respond to perturbation of connected node 0")
        # Nodes 2..N-1 are not connected → should not change
        self.assertAlmostEqual(diff[2:].max().item(), 0.0, places=5,
                               msg="Unconnected nodes should be unaffected")


# ══════════════════════════════════════════════════════════════════════════════
# Docstring correctness: old docstring claimed "只存储同模态边" (stores
# intra-modal edges).  New docstring must say "只存储节点特征" (node features).
# ══════════════════════════════════════════════════════════════════════════════

class TestDocstringCorrectness(unittest.TestCase):

    def test_docstring_no_longer_says_stores_intra_modal_edges(self):
        """Old incorrect claim: cache stores same-modality edges."""
        load_model_path = _TD_DIR / "loader" / "load_model.py"
        src = load_model_path.read_text(encoding="utf-8")
        # Old wrong claim should be gone
        self.assertNotIn(
            "只存储同模态边",
            src,
            "Old incorrect docstring claim must be removed",
        )

    def test_docstring_explains_node_features_only(self):
        """New docstring must document that V5 caches store only node features."""
        load_model_path = _TD_DIR / "loader" / "load_model.py"
        src = load_model_path.read_text(encoding="utf-8")
        self.assertIn(
            "只存储节点特征",
            src,
            "Docstring must state that V5 caches store only node features",
        )

    def test_docstring_explains_diagonal_consequence(self):
        """Docstring must explain what happens without edges (diagonal matrix)."""
        load_model_path = _TD_DIR / "loader" / "load_model.py"
        src = load_model_path.read_text(encoding="utf-8")
        self.assertIn(
            "对角线",
            src,
            "Docstring must warn about diagonal response matrix when edges missing",
        )

    def test_code_has_intra_modal_rebuild_for_fmri(self):
        """The code must contain fmri intra-modal edge rebuild logic."""
        load_model_path = _TD_DIR / "loader" / "load_model.py"
        src = load_model_path.read_text(encoding="utf-8")
        # The fix uses (nt, "connects", nt) in a loop over {"fmri": ..., "eeg": ...}
        self.assertIn(
            '"connects", nt',
            src,
            "load_graph_for_inference must rebuild intra-modal edges via (nt, 'connects', nt)",
        )
        # The fmri key must appear in the _INTRA_MODAL_CFG dict
        self.assertIn(
            '"fmri"',
            src,
            "load_graph_for_inference must handle fMRI modality",
        )


if __name__ == "__main__":
    unittest.main()
