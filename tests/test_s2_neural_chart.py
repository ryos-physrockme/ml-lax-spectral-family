import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "07_s2_neural_chart" / "train.py"
spec = importlib.util.spec_from_file_location("s2_neural_chart_train_for_test", SCRIPT)
assert spec is not None and spec.loader is not None
chart = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chart
spec.loader.exec_module(chart)


def test_complex_real_round_trip_for_chart_coefficients():
    coefficients = np.asarray(
        [[1.0 + 0.2j, 0.7 - 0.1j, 0.9 + 0.4j, 1.4 - 0.3j]],
        dtype=np.complex128,
    )
    raw = chart.complex_coefficients_to_real(coefficients)
    reconstructed = raw[:, :4] + 1j * raw[:, 4:]
    np.testing.assert_allclose(reconstructed, coefficients)


def test_chart_rank_diagnostic_distinguishes_independent_and_parallel_tangents():
    independent = torch.zeros(1, 8, 2, dtype=torch.float64)
    independent[0, 0, 0] = 1.0
    independent[0, 1, 1] = 1.0
    _, sine_squared_independent, singular_values_independent = chart.jacobian_geometry(independent)
    assert float(sine_squared_independent[0]) == 1.0
    torch.testing.assert_close(
        singular_values_independent[0],
        torch.ones(2, dtype=torch.float64),
    )

    parallel = torch.zeros(1, 8, 2, dtype=torch.float64)
    parallel[0, 0, 0] = 1.0
    parallel[0, 0, 1] = 1.0
    _, sine_squared_parallel, singular_values_parallel = chart.jacobian_geometry(parallel)
    assert float(sine_squared_parallel[0]) == 0.0
    assert float(singular_values_parallel[0, 1]) < 1.0e-14
