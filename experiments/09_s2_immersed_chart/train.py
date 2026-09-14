from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


previous = load_module(
    "s2_transverse_tangent_for_experiment09",
    ROOT / "experiments" / "08_s2_transverse_tangent" / "train.py",
)
chart = previous.chart
discovery = previous.discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def curriculum_half_width(step: int, schedule: list[dict]) -> float:
    for entry in schedule:
        if step <= int(entry["until_step"]):
            return float(entry["half_width"])
    return float(schedule[-1]["half_width"])


def immersion_losses(
    singular_values: torch.Tensor,
    ratio_minimum: float,
    singular_value_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_max = singular_values[:, 0]
    sigma_min = singular_values[:, -1]
    ratio = sigma_min / torch.clamp(sigma_max, min=1.0e-30)
    ratio_loss = torch.mean(torch.relu(ratio_minimum - ratio) ** 2)
    floor_loss = torch.mean(torch.relu(singular_value_floor - sigma_min) ** 2)
    return ratio_loss, floor_loss, ratio


def calculate_objective(
    model,
    coordinates: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_coordinates_all: torch.Tensor,
    anchor_targets_all: torch.Tensor,
    tangent_s_all: torch.Tensor,
    tangent_t_all: torch.Tensor,
    flatness_data,
    cfg: dict,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scfg = cfg["sampling"]
    tcfg = cfg["training"]

    raw, jacobian = chart.output_and_jacobian(
        model,
        coordinates.clone(),
        create_graph=create_graph,
    )
    coefficients = chart.real_output_to_complex(raw)
    flatness_loss = torch.mean(
        discovery.component_normalized_loss(
            coefficients,
            flatness_data,
            float(scfg["denominator_epsilon"]),
        )
    )

    _, _, singular_values = chart.jacobian_geometry(jacobian)
    ratio_loss, floor_loss, sigma_ratio = immersion_losses(
        singular_values,
        float(tcfg["rank_ratio_minimum"]),
        float(tcfg["singular_value_floor"]),
    )

    anchor_coordinates = anchor_coordinates_all[anchor_indices]
    anchor_prediction, anchor_jacobian = chart.output_and_jacobian(
        model,
        anchor_coordinates.clone(),
        create_graph=create_graph,
    )
    anchor_loss = torch.mean(
        (anchor_prediction - anchor_targets_all[anchor_indices]) ** 2
    )
    frame_loss = torch.mean(
        torch.sum(
            (anchor_jacobian[..., 0] - tangent_s_all[anchor_indices]) ** 2,
            dim=-1,
        )
        + torch.sum(
            (anchor_jacobian[..., 1] - tangent_t_all[anchor_indices]) ** 2,
            dim=-1,
        )
    )

    total = (
        float(tcfg["flatness_weight"]) * flatness_loss
        + float(tcfg["anchor_weight"]) * anchor_loss
        + float(tcfg["frame_weight"]) * frame_loss
        + float(tcfg["rank_ratio_weight"]) * ratio_loss
        + float(tcfg["singular_value_floor_weight"]) * floor_loss
    )
    metrics = {
        "flatness_loss": flatness_loss,
        "anchor_loss": anchor_loss,
        "frame_loss": frame_loss,
        "rank_ratio_loss": ratio_loss,
        "singular_value_floor_loss": floor_loss,
        "median_sigma_ratio": torch.median(sigma_ratio),
        "median_sigma_min": torch.median(singular_values[:, -1]),
        "median_sigma_max": torch.median(singular_values[:, 0]),
    }
    return total, metrics


def detached_metrics(total: torch.Tensor, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    output = {"total_loss": float(total.detach().cpu())}
    output.update({key: float(value.detach().cpu()) for key, value in metrics.items()})
    return output


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    anchors = chart.load_continuation_trace(
        ROOT / cfg["input_trace"],
        int(cfg["anchors"]["n_points"]),
    )
    scfg = cfg["sampling"]
    tcfg = cfg["training"]
    dcfg = cfg["chart_domain"]
    acfg = cfg["anchors"]
    ncfg = cfg["network"]

    jacobian_data_torch = discovery.sample_onshell_data(
        int(scfg["jacobian_samples"]),
        np.random.default_rng(seed + 1),
        torch.device("cpu"),
        float(scfg["theta_margin"]),
    )
    frames = previous.construct_tangent_frames(
        anchors,
        discovery.data_as_numpy(jacobian_data_torch),
    )

    flatness_data = discovery.sample_onshell_data(
        int(scfg["flatness_samples"]),
        np.random.default_rng(seed + 101),
        device,
        float(scfg["theta_margin"]),
    )
    validation_flatness_data = discovery.sample_onshell_data(
        int(scfg["validation_flatness_samples"]),
        np.random.default_rng(seed + 151),
        device,
        float(scfg["theta_margin"]),
    )
    evaluation_data = discovery.sample_onshell_data(
        int(scfg["evaluation_flatness_samples"]),
        np.random.default_rng(seed + 202),
        device,
        float(scfg["theta_margin"]),
    )

    s_min = float(np.min(anchors.s))
    s_max = float(np.max(anchors.s))
    t_min = float(dcfg["transverse_min"])
    t_max = float(dcfg["transverse_max"])
    center_index = int(np.argmin(np.abs(anchors.s)))
    initial_bias = chart.complex_coefficients_to_real(
        anchors.coefficients[center_index:center_index + 1]
    )[0]
    model = chart.ChartNetwork(
        s_scale=max(abs(s_min), abs(s_max), 1.0),
        t_scale=max(abs(t_min), abs(t_max), 1.0),
        hidden_dim=int(ncfg["hidden_dim"]),
        hidden_layers=int(ncfg["hidden_layers"]),
        initial_bias=initial_bias,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]))

    anchor_coordinates_all = torch.as_tensor(
        np.column_stack([anchors.s, np.zeros_like(anchors.s)]),
        dtype=torch.float64,
        device=device,
    )
    anchor_targets_all = torch.as_tensor(
        chart.complex_coefficients_to_real(anchors.coefficients),
        dtype=torch.float64,
        device=device,
    )
    tangent_s_all = torch.as_tensor(frames["tangent_s"], dtype=torch.float64, device=device)
    tangent_t_all = torch.as_tensor(frames["tangent_t"], dtype=torch.float64, device=device)

    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(seed + 17)
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(seed + 19)
    rng = np.random.default_rng(seed + 31)

    validation_coordinates = chart.sample_chart_coordinates(
        int(tcfg["validation_latent_points"]),
        s_min,
        s_max,
        t_min,
        t_max,
        validation_generator,
        device,
    ).detach()
    all_anchor_indices = torch.arange(anchors.s.size, dtype=torch.long, device=device)

    history: list[dict] = []
    best_score = float("inf")
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())

    schedule = list(tcfg["transverse_curriculum"])
    for step in range(1, int(tcfg["steps"]) + 1):
        half_width = curriculum_half_width(step, schedule)
        coordinates = chart.sample_chart_coordinates(
            int(tcfg["latent_batch_size"]),
            s_min,
            s_max,
            -half_width,
            half_width,
            train_generator,
            device,
        )
        anchor_indices_np = rng.integers(
            0,
            anchors.s.size,
            size=int(acfg["batch_size"]),
        )
        anchor_indices = torch.as_tensor(anchor_indices_np, dtype=torch.long, device=device)

        total, metrics = calculate_objective(
            model,
            coordinates,
            anchor_indices,
            anchor_coordinates_all,
            anchor_targets_all,
            tangent_s_all,
            tangent_t_all,
            flatness_data,
            cfg,
            create_graph=True,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if step == 1 or step % int(tcfg["log_every"]) == 0 or step == int(tcfg["steps"]):
            validation_total, validation_metrics = calculate_objective(
                model,
                validation_coordinates,
                all_anchor_indices,
                anchor_coordinates_all,
                anchor_targets_all,
                tangent_s_all,
                tangent_t_all,
                validation_flatness_data,
                cfg,
                create_graph=False,
            )
            validation_score = float(validation_total.detach().cpu())
            if validation_score < best_score:
                best_score = validation_score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

            record = {
                "step": step,
                "training_half_width": half_width,
                "training": detached_metrics(total, metrics),
                "validation": detached_metrics(validation_total, validation_metrics),
                "best_validation_total": best_score,
                "best_step": best_step,
            }
            history.append(record)
            print(json.dumps(record))

    model.load_state_dict(best_state)

    summary, arrays = chart.evaluate_variant(
        model,
        cfg,
        anchors,
        evaluation_data,
        device,
    )
    frame_summary, frame_arrays = previous.evaluate_frame_alignment(
        model,
        anchors,
        frames,
        device,
    )
    summary.update(frame_summary)
    singular_values_grid = arrays["jacobian_singular_values"]
    summary.update(
        {
            "best_step": int(best_step),
            "best_validation_total": float(best_score),
            "jacobian_sigma_min_median": float(np.median(singular_values_grid[..., -1])),
            "jacobian_sigma_min_minimum": float(np.min(singular_values_grid[..., -1])),
            "jacobian_sigma_max_median": float(np.median(singular_values_grid[..., 0])),
            "jacobian_sigma_max_maximum": float(np.max(singular_values_grid[..., 0])),
        }
    )
    arrays.update(frame_arrays)
    arrays["path_tangent_projection_cosine"] = frames["path_projection_cosine"]

    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(output_dir / "evaluation.npz", **arrays)
    torch.save(model.state_dict(), output_dir / "model_best.pt")

    full_summary = {
        "tangent_frame_construction": {
            "minimum_path_tangent_projection_cosine": float(
                np.min(frames["path_projection_cosine"])
            ),
            "median_path_tangent_projection_cosine": float(
                np.median(frames["path_projection_cosine"])
            ),
        },
        "chart_coordinate": {
            "definition": "z=s+i t; s is signed pseudo-arclength on the Experiment 06 anchor path",
            "canonical_spectral_parameter_claimed": False,
        },
        "immersed_chart": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(full_summary, indent=2))
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
