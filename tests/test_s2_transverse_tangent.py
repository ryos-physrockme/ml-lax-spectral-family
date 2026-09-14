import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "08_s2_transverse_tangent" / "train.py"
spec = importlib.util.spec_from_file_location("s2_transverse_tangent_train_for_test", SCRIPT)
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def test_orthogonal_direction_is_in_plane_and_normalized():
    basis = np.zeros((2, 8), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    first = basis[0]
    second = experiment.orthogonal_direction_in_plane(basis, first)
    np.testing.assert_allclose(np.linalg.norm(second), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.dot(first, second), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(second[2:], 0.0, atol=1.0e-14)


def test_logarithmic_rank_barrier_penalizes_parallel_tangents_more():
    independent = torch.tensor([1.0], dtype=torch.float64)
    nearly_parallel = torch.tensor([1.0e-4], dtype=torch.float64)
    independent_loss = experiment.logarithmic_rank_barrier(independent, 1.0e-6)
    parallel_loss = experiment.logarithmic_rank_barrier(nearly_parallel, 1.0e-6)
    assert float(parallel_loss) > float(independent_loss)
