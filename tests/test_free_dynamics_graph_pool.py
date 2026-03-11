import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).parent.parent
_BD_DIR = _REPO_ROOT / "brain_dynamics"
if str(_BD_DIR) not in sys.path:
    sys.path.insert(0, str(_BD_DIR))


def _import_free_dynamics_with_stubs():
    """Import experiments.free_dynamics without requiring torch/pyg runtime."""
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )

    sim_pkg = types.ModuleType("simulator")
    sim_mod = types.ModuleType("simulator.brain_dynamics_simulator")
    sim_mod.BrainDynamicsSimulator = object

    with patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "simulator": sim_pkg,
            "simulator.brain_dynamics_simulator": sim_mod,
        },
    ):
        if "experiments.free_dynamics" in sys.modules:
            del sys.modules["experiments.free_dynamics"]
        return importlib.import_module("experiments.free_dynamics")


class _FakeNode:
    def __init__(self, n: int, t: int = 5):
        self.x = types.SimpleNamespace(shape=(n, t, 1))


class _FakeBaseGraph(dict):
    @property
    def node_types(self):
        return list(self.keys())


class _FakeSimulator:
    def __init__(self):
        self.n_regions = 253  # joint total (fmri + eeg)
        self.base_graph = _FakeBaseGraph(
            fmri=_FakeNode(190),
            eeg=_FakeNode(63),
        )


class TestGraphPoolPrimaryCount(unittest.TestCase):
    def test_build_graph_pool_uses_primary_modality_node_count(self):
        fd = _import_free_dynamics_with_stubs()
        sim = _FakeSimulator()
        paths = [Path("main.pt"), Path("extra.pt")]

        with patch.object(fd, "_load_graph_x_only", return_value=None) as mocked_loader:
            fd._build_graph_pool(paths, sim, "fmri")

        self.assertEqual(mocked_loader.call_count, 1)
        _, _, n_regions_primary, nt_primary = mocked_loader.call_args[0]
        self.assertEqual(nt_primary, "fmri")
        self.assertEqual(
            n_regions_primary,
            190,
            "Must validate extra graph node count against primary modality (fmri), "
            "not joint total (fmri+eeg).",
        )


if __name__ == "__main__":
    unittest.main()
