import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "09_s2_immersed_chart" / "train.py"
spec = importlib.util.spec_from_file_location("s2_immersed_chart_train_for_test", SCRIPT)
assert spec is not None and spec.loader is not None
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def test_transverse_width_curriculum():
    schedule = [
        {"until_step": 10, "half_width": 0.05},
        {"until_step": 20, "half_width": 0.15},
        {"until_step": 30, "half_width": 0.50},
    ]
    assert experiment.curriculum_half_width(1, schedule) == 0.05
    assert experiment.curriculum_half_width(10, schedule) == 0.05
    assert experiment.curriculum_half_width(11, schedule) == 0.15
    assert experiment.curriculum_half_width(30, schedule) == 0.50
    assert experiment.curriculum_half_width(31, schedule) == 0.50


def test_immersion_penalties_only_act_below_lower_bounds():
    singular_values_good = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    ratio_loss, floor_loss, ratio = experiment.immersion_losses(
        singular_values_good,
        ratio_minimum=0.1,
        singular_value_floor=0.05,
    )
    assert float(ratio_loss) == 0.0
    assert float(floor_loss) == 0.0
    assert float(ratio[0]) == 0.5

    singular_values_bad = torch.tensor([[1.0, 0.01]], dtype=torch.float64)
    ratio_loss, floor_loss, _ = experiment.immersion_losses(
        singular_values_bad,
        ratio_minimum=0.1,
        singular_value_floor=0.05,
    )
    assert float(ratio_loss) > 0.0
    assert float(floor_loss) > 0.0
