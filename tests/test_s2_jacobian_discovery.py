from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "05_s2_jacobian_family_discovery"
    / "train.py"
)
spec = importlib.util.spec_from_file_location("s2_discovery_train", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_family_has_zero_component_normalized_loss() -> None:
    rng = np.random.default_rng(10)
    data = module.sample_onshell_data(256, rng, torch.device("cpu"), 0.35)
    b = 0.7 + 0.4j
    parameters = torch.tensor(
        [[1.0 + 0.0j, b, 1.0 + 0.0j, 1.0 / b]],
        dtype=torch.complex128,
    )
    loss = module.component_normalized_loss(parameters, data, 1.0e-10)
    assert float(loss[0]) < 1.0e-24


def test_exact_family_jacobian_has_two_real_null_directions() -> None:
    rng = np.random.default_rng(11)
    data_torch = module.sample_onshell_data(1024, rng, torch.device("cpu"), 0.35)
    data = module.data_as_numpy(data_torch)
    b = 1.2 - 0.3j
    point = np.asarray([1.0 + 0.0j, b, 1.0 + 0.0j, 1.0 / b])
    jac = module.realified_curvature_jacobian(point, data)
    jac = jac / np.linalg.norm(jac, axis=0, keepdims=True)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    assert singular_values[5] > 1.0e-3
    assert singular_values[6] < 1.0e-12
    assert singular_values[7] < 1.0e-12
