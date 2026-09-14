from __future__ import annotations

import argparse
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


discovery = load_module(
    "s2_discovery_train_for_experiment08",
    ROOT / "experiments" / "05_s2_jacobian_family_discovery" / "train.py",
)
continuation = load_module(
    "s2_continuation_for_experiment08",
    ROOT / "experiments" / "06_s2_pseudo_arclength_continuation" / "continue_family.py",
)
chart = load_module(
    "s2_neural_chart_for_experiment08",
    ROOT / "experiments" / "07_s2_neural_chart" / "train.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def coefficient_array_to_real(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    return np.concatenate([coefficients.real, coefficients.imag], axis=-1)


def orthogonal_direction_in_plane(
    basis: np.ndarray,
    first_direction: np.ndarray,
) -> np.ndarray:
    """Return a unit vector in span(basis) orthogonal to first_direction."""

    first = np.asarray(first_direction, dtype=np.float64)
    first = first / np.linalg.norm(first)
    for row in np.asarray(basis, dtype=np.float64):
        candidate = row - np.dot(row, first) * first
        norm = np.linalg.norm(candidate)
        if norm > 1.0e-12:
            return candidate / norm
    raise RuntimeError("Could not construct the second tangent direction")


def construct_tangent_frames(
    anchors,
    jacobian_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct an oriented orthonormal tangent frame on the anchor line.

    The first direction follows the pseudo-arclength path after projection into
    the two-dimensional flatness-Jacobian null space.  The second direction is
    the orthogonal direction inside that same null space.  Its sign is fixed by
    continuity along the ordered anchor points.
    """

    real_points = coefficient_array_to_real(anchors.coefficients)
    raw_path_tangent = np.gradient(real_points, anchors.s, axis=0, edge_order=2)

    tangent_s = np.empty_like(real_points)
    tangent_t = np.empty_like(real_points)
    projection_cosine = np.empty(anchors.s.size, dtype=np.float64)
    singular_values = np.empty((anchors.s.size, 8), dtype=np.float64)

    previous_transverse = None
    for index, point in enumerate(anchors.coefficients):
        basis, svals = continuation.physical_null_basis(point, jacobian_data)
        singular_values[index] = svals

        raw = raw_path_tangent[index]
        raw_norm = np.linalg.norm(raw)
        if raw_norm < 1.0e-14:
            raise RuntimeError("The finite-difference path tangent is numerically zero")
        raw_unit = raw / raw_norm

        projected = basis.T @ (basis @ raw_unit)
        projected_norm = np.linalg.norm(projected)
        if projected_norm < 1.0e-14:
            raise RuntimeError("The continuation tangent has no component in the Jacobian null space")
        first = projected / projected_norm
        projection_cosine[index] = abs(float(np.dot(first, raw_unit)))

        second = orthogonal_direction_in_plane(basis, first)
        if previous_transverse is None:
            largest = int(np.argmax(np.abs(second)))
            if second[largest] < 0.0:
                second = -second
        elif np.dot(second, previous_transverse) < 0.0:
            second = -second

        tangent_s[index] = first
        tangent_t[index] = second
        previous_transverse = second

    return {
        "tangent_s": tangent_s,
        "tangent_t": tangent_t,
        "path_projection_cosine": projection_cosine,
        "jacobian_singular_values": singular_values,
    }


def logarithmic_rank_barrier(
    sine_squared: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return -torch.mean(torch.log(torch.clamp(sine_squared, min=0.0) + epsilon))


def train_variant(
    name: str,
    variant: dict,
    cfg: dict,
    anchors,
    frames: dict[str, np.ndarray],
    flatness_data,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 17)
    rng = np.random.default_rng(seed + 31)

    ncfg = cfg["network"]
    dcfg = cfg["chart_domain"]
    tcfg = cfg["training"]
    acfg = cfg["anchors"]
    scfg = cfg["sampling"]

    s_min = float(np.min(anchors.s))
    s_max = float(np.max(anchors.s))
    t_min = float(dcfg["transverse_min"])
    t_max = float(dcfg["transverse_max"])
    s_scale = max(abs(s_min), abs(s_max), 1.0)
    t_scale = max(abs(t_min), abs(t_max), 1.0)

    center_index = int(np.argmin(np.abs(anchors.s)))
    initial_bias = chart.complex_coefficients_to_real(
        anchors.coefficients[center_index:center_index + 1]
    )[0]
    model = chart.ChartNetwork(
        s_scale=s_scale,
        t_scale=t_scale,
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

    history: list[dict] = []
    for step in range(1, int(tcfg["steps"]) + 1):
        coordinates = chart.sample_chart_coordinates(
            int(tcfg["latent_batch_size"]),
            s_min,
            s_max,
            t_min,
            t_max,
            generator,
            device,
        )
        raw, jacobian = chart.output_and_jacobian(model, coordinates, create_graph=True)
        coefficients = chart.real_output_to_complex(raw)
        flatness_loss = torch.mean(
            discovery.component_normalized_loss(
                coefficients,
                flatness_data,
                float(scfg["denominator_epsilon"]),
            )
        )

        metric_diagonal, sine_squared, singular_values = chart.jacobian_geometry(jacobian)
        derivative_norm_loss = torch.mean((metric_diagonal - 1.0) ** 2)
        barrier_loss = logarithmic_rank_barrier(
            sine_squared,
            float(tcfg["barrier_epsilon"]),
        )

        anchor_indices = rng.integers(0, anchors.s.size, size=int(acfg["batch_size"]))
        anchor_indices_t = torch.as_tensor(anchor_indices, dtype=torch.long, device=device)
        anchor_coordinates = anchor_coordinates_all[anchor_indices_t]
        anchor_prediction, anchor_jacobian = chart.output_and_jacobian(
            model,
            anchor_coordinates,
            create_graph=True,
        )
        anchor_loss = torch.mean(
            (anchor_prediction - anchor_targets_all[anchor_indices_t]) ** 2
        )
        frame_loss = torch.mean(
            torch.sum(
                (anchor_jacobian[..., 0] - tangent_s_all[anchor_indices_t]) ** 2,
                dim=-1,
            )
            + torch.sum(
                (anchor_jacobian[..., 1] - tangent_t_all[anchor_indices_t]) ** 2,
                dim=-1,
            )
        )

        total_loss = (
            float(tcfg["flatness_weight"]) * flatness_loss
            + float(tcfg["anchor_weight"]) * anchor_loss
            + float(tcfg["derivative_norm_weight"]) * derivative_norm_loss
            + float(variant["barrier_weight"]) * barrier_loss
            + float(variant["frame_weight"]) * frame_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % int(tcfg["log_every"]) == 0 or step == int(tcfg["steps"]):
            with torch.no_grad():
                sigma_ratio = singular_values[:, -1] / torch.clamp(
                    singular_values[:, 0], min=1.0e-30
                )
                record = {
                    "step": step,
                    "total_loss": float(total_loss.detach().cpu()),
                    "flatness_loss": float(flatness_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "derivative_norm_loss": float(derivative_norm_loss.detach().cpu()),
                    "barrier_loss": float(barrier_loss.detach().cpu()),
                    "frame_loss": float(frame_loss.detach().cpu()),
                    "median_sine_squared": float(torch.median(sine_squared).detach().cpu()),
                    "median_jacobian_sigma_ratio": float(torch.median(sigma_ratio).detach().cpu()),
                }
            history.append(record)
            print(json.dumps({"variant": name, **record}))

    return model, history


def evaluate_frame_alignment(
    model,
    anchors,
    frames: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    coordinates = torch.as_tensor(
        np.column_stack([anchors.s, np.zeros_like(anchors.s)]),
        dtype=torch.float64,
        device=device,
    )
    _, jacobian = chart.output_and_jacobian(model, coordinates, create_graph=False)
    jacobian_np = jacobian.detach().cpu().numpy()

    predicted_s = jacobian_np[..., 0]
    predicted_t = jacobian_np[..., 1]
    target_s = frames["tangent_s"]
    target_t = frames["tangent_t"]

    predicted_s_unit = predicted_s / np.maximum(
        np.linalg.norm(predicted_s, axis=-1, keepdims=True), 1.0e-300
    )
    predicted_t_unit = predicted_t / np.maximum(
        np.linalg.norm(predicted_t, axis=-1, keepdims=True), 1.0e-300
    )
    cosine_s = np.sum(predicted_s_unit * target_s, axis=-1)
    cosine_t = np.sum(predicted_t_unit * target_t, axis=-1)

    line_singular_values = np.linalg.svd(jacobian_np, compute_uv=False)
    line_sigma_ratio = line_singular_values[:, -1] / np.maximum(
        line_singular_values[:, 0], 1.0e-300
    )

    summary = {
        "anchor_tangent_s_cosine_median": float(np.median(cosine_s)),
        "anchor_tangent_s_cosine_min": float(np.min(cosine_s)),
        "anchor_tangent_t_cosine_median": float(np.median(cosine_t)),
        "anchor_tangent_t_cosine_min": float(np.min(cosine_t)),
        "anchor_line_sigma_ratio_median": float(np.median(line_sigma_ratio)),
        "anchor_line_sigma_ratio_min": float(np.min(line_sigma_ratio)),
    }
    arrays = {
        "anchor_frame_s": anchors.s,
        "predicted_tangent_s": predicted_s,
        "predicted_tangent_t": predicted_t,
        "target_tangent_s": target_s,
        "target_tangent_t": target_t,
        "tangent_s_cosine": cosine_s,
        "tangent_t_cosine": cosine_t,
        "anchor_line_sigma_ratio": line_sigma_ratio,
    }
    return summary, arrays


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg["seed"])

    anchors = chart.load_continuation_trace(
        ROOT / cfg["input_trace"],
        int(cfg["anchors"]["n_points"]),
    )

    scfg = cfg["sampling"]
    jacobian_data_torch = discovery.sample_onshell_data(
        int(scfg["jacobian_samples"]),
        np.random.default_rng(seed + 1),
        torch.device("cpu"),
        float(scfg["theta_margin"]),
    )
    frames = construct_tangent_frames(
        anchors,
        discovery.data_as_numpy(jacobian_data_torch),
    )

    flatness_data = discovery.sample_onshell_data(
        int(scfg["flatness_samples"]),
        np.random.default_rng(seed + 101),
        device,
        float(scfg["theta_margin"]),
    )
    evaluation_data = discovery.sample_onshell_data(
        int(scfg["evaluation_flatness_samples"]),
        np.random.default_rng(seed + 202),
        device,
        float(scfg["theta_margin"]),
    )

    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_construction_summary = {
        "minimum_path_tangent_projection_cosine": float(
            np.min(frames["path_projection_cosine"])
        ),
        "median_path_tangent_projection_cosine": float(
            np.median(frames["path_projection_cosine"])
        ),
        "minimum_jacobian_s6_over_s7_on_anchors": float(
            np.min(
                frames["jacobian_singular_values"][:, 5]
                / np.maximum(frames["jacobian_singular_values"][:, 6], 1.0e-300)
            )
        ),
    }

    all_summary: dict[str, dict] = {
        "tangent_frame_construction": frame_construction_summary,
        "chart_coordinate": {
            "definition": "z=s+i t; s is signed pseudo-arclength on the Experiment 06 anchor path",
            "canonical_spectral_parameter_claimed": False,
        },
    }

    for variant in cfg["variants"]:
        name = str(variant["name"])
        model, history = train_variant(
            name=name,
            variant=variant,
            cfg=cfg,
            anchors=anchors,
            frames=frames,
            flatness_data=flatness_data,
            device=device,
            seed=seed,
        )
        summary, arrays = chart.evaluate_variant(
            model,
            cfg,
            anchors,
            evaluation_data,
            device,
        )
        frame_summary, frame_arrays = evaluate_frame_alignment(
            model,
            anchors,
            frames,
            device,
        )
        summary.update(frame_summary)
        summary["barrier_weight"] = float(variant["barrier_weight"])
        summary["frame_weight"] = float(variant["frame_weight"])
        all_summary[name] = summary

        arrays.update(frame_arrays)
        arrays["path_tangent_projection_cosine"] = frames["path_projection_cosine"]
        (output_dir / f"history_{name}.json").write_text(json.dumps(history, indent=2))
        np.savez_compressed(output_dir / f"evaluation_{name}.npz", **arrays)
        torch.save(model.state_dict(), output_dir / f"model_{name}.pt")
        print(json.dumps({"variant": name, "evaluation": summary}, indent=2))

    (output_dir / "summary.json").write_text(json.dumps(all_summary, indent=2))
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
