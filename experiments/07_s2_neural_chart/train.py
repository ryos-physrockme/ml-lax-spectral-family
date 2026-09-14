from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_TRAIN = ROOT / "experiments" / "05_s2_jacobian_family_discovery" / "train.py"
spec = importlib.util.spec_from_file_location("s2_discovery_train_for_chart", DISCOVERY_TRAIN)
assert spec is not None and spec.loader is not None
discovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = discovery
spec.loader.exec_module(discovery)


@dataclass(frozen=True)
class AnchorData:
    s: np.ndarray
    coefficients: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def pair_to_complex(pair: list[float]) -> complex:
    return complex(float(pair[0]), float(pair[1]))


def load_continuation_trace(path: Path, n_anchors: int) -> AnchorData:
    rows = json.loads(path.read_text())
    s_raw = np.asarray([float(row["signed_arclength"]) for row in rows], dtype=np.float64)
    coeff_raw = np.asarray(
        [
            [
                pair_to_complex(row["a"]),
                pair_to_complex(row["b"]),
                pair_to_complex(row["c"]),
                pair_to_complex(row["d"]),
            ]
            for row in rows
        ],
        dtype=np.complex128,
    )

    order = np.argsort(s_raw)
    s_raw = s_raw[order]
    coeff_raw = coeff_raw[order]

    # The continuation path may contain numerically repeated arclength values.
    # Keep the first occurrence so that interpolation is well-defined.
    s_unique, unique_indices = np.unique(s_raw, return_index=True)
    coeff_unique = coeff_raw[unique_indices]

    s_anchor = np.linspace(float(s_unique[0]), float(s_unique[-1]), n_anchors)
    coeff_anchor = np.empty((n_anchors, 4), dtype=np.complex128)
    for column in range(4):
        coeff_anchor[:, column] = np.interp(
            s_anchor, s_unique, coeff_unique[:, column].real
        ) + 1j * np.interp(s_anchor, s_unique, coeff_unique[:, column].imag)
    return AnchorData(s=s_anchor, coefficients=coeff_anchor)


def complex_coefficients_to_real(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    return np.concatenate([coefficients.real, coefficients.imag], axis=-1)


def real_output_to_complex(raw: torch.Tensor) -> torch.Tensor:
    return torch.complex(raw[..., :4], raw[..., 4:])


class ChartNetwork(nn.Module):
    """Map two real chart coordinates to four complex Lax coefficients.

    The physical input coordinates are retained outside the network.  Internal
    rescaling only improves conditioning; derivatives used in the loss are
    always taken with respect to the physical coordinates (s,t).
    """

    def __init__(
        self,
        s_scale: float,
        t_scale: float,
        hidden_dim: int,
        hidden_layers: int,
        initial_bias: np.ndarray,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")
        self.register_buffer("input_scale", torch.tensor([s_scale, t_scale], dtype=torch.float64))

        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        output_layer = nn.Linear(hidden_dim, 8)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)
        self.double()

        with torch.no_grad():
            output_layer.weight.mul_(0.02)
            output_layer.bias.copy_(torch.as_tensor(initial_bias, dtype=torch.float64))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.net(coordinates / self.input_scale)


def output_and_jacobian(
    model: nn.Module,
    coordinates: torch.Tensor,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = coordinates.requires_grad_(True)
    output = model(coordinates)
    derivatives = []
    for component in range(output.shape[-1]):
        gradient = torch.autograd.grad(
            output[:, component].sum(),
            coordinates,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        derivatives.append(gradient)
    jacobian = torch.stack(derivatives, dim=1)
    return output, jacobian


def jacobian_geometry(jacobian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tangent_s = jacobian[..., 0]
    tangent_t = jacobian[..., 1]
    g_ss = torch.sum(tangent_s * tangent_s, dim=-1)
    g_tt = torch.sum(tangent_t * tangent_t, dim=-1)
    g_st = torch.sum(tangent_s * tangent_t, dim=-1)
    denominator = torch.clamp(g_ss * g_tt, min=1.0e-30)
    sine_squared = torch.clamp((g_ss * g_tt - g_st * g_st) / denominator, min=0.0, max=1.0)
    singular_values = torch.linalg.svdvals(jacobian)
    return torch.stack([g_ss, g_tt], dim=-1), sine_squared, singular_values


def sample_chart_coordinates(
    batch_size: int,
    s_min: float,
    s_max: float,
    t_min: float,
    t_max: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    unit = torch.rand(batch_size, 2, dtype=torch.float64, generator=generator, device=device)
    coordinates = torch.empty_like(unit)
    coordinates[:, 0] = s_min + (s_max - s_min) * unit[:, 0]
    coordinates[:, 1] = t_min + (t_max - t_min) * unit[:, 1]
    return coordinates


def train_variant(
    name: str,
    use_rank_penalty: bool,
    cfg: dict,
    anchors: AnchorData,
    flatness_data,
    device: torch.device,
    seed: int,
) -> tuple[ChartNetwork, list[dict]]:
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
    initial_bias = complex_coefficients_to_real(anchors.coefficients[center_index:center_index + 1])[0]
    model = ChartNetwork(
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
        complex_coefficients_to_real(anchors.coefficients),
        dtype=torch.float64,
        device=device,
    )

    history: list[dict] = []
    for step in range(1, int(tcfg["steps"]) + 1):
        coordinates = sample_chart_coordinates(
            int(tcfg["latent_batch_size"]),
            s_min,
            s_max,
            t_min,
            t_max,
            generator,
            device,
        )
        raw, jacobian = output_and_jacobian(model, coordinates, create_graph=True)
        coefficients = real_output_to_complex(raw)

        flatness_per_point = discovery.component_normalized_loss(
            coefficients,
            flatness_data,
            float(scfg["denominator_epsilon"]),
        )
        flatness_loss = torch.mean(flatness_per_point)

        anchor_indices = rng.integers(0, anchors.s.size, size=int(acfg["batch_size"]))
        anchor_indices_t = torch.as_tensor(anchor_indices, dtype=torch.long, device=device)
        anchor_prediction = model(anchor_coordinates_all[anchor_indices_t])
        anchor_loss = torch.mean((anchor_prediction - anchor_targets_all[anchor_indices_t]) ** 2)

        metric_diagonal, sine_squared, singular_values = jacobian_geometry(jacobian)
        derivative_norm_loss = torch.mean((metric_diagonal - 1.0) ** 2)

        minimum_sine_squared = float(tcfg["minimum_sine_squared"])
        if use_rank_penalty:
            rank_loss = torch.mean(torch.relu(minimum_sine_squared - sine_squared) ** 2)
        else:
            rank_loss = torch.zeros((), dtype=torch.float64, device=device)

        total_loss = (
            float(tcfg["flatness_weight"]) * flatness_loss
            + float(tcfg["anchor_weight"]) * anchor_loss
            + float(tcfg["derivative_norm_weight"]) * derivative_norm_loss
            + float(tcfg["rank_weight"]) * rank_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % int(tcfg["log_every"]) == 0 or step == int(tcfg["steps"]):
            with torch.no_grad():
                sigma_ratio = singular_values[:, -1] / torch.clamp(singular_values[:, 0], min=1.0e-30)
                record = {
                    "step": step,
                    "total_loss": float(total_loss.detach().cpu()),
                    "flatness_loss": float(flatness_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "derivative_norm_loss": float(derivative_norm_loss.detach().cpu()),
                    "rank_loss": float(rank_loss.detach().cpu()),
                    "median_sine_squared": float(torch.median(sine_squared).detach().cpu()),
                    "median_jacobian_sigma_ratio": float(torch.median(sigma_ratio).detach().cpu()),
                }
            history.append(record)
            print(json.dumps({"variant": name, **record}))

    return model, history


def evaluate_variant(
    model: ChartNetwork,
    cfg: dict,
    anchors: AnchorData,
    evaluation_data,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    dcfg = cfg["chart_domain"]
    scfg = cfg["sampling"]

    s_values = np.linspace(
        float(np.min(anchors.s)),
        float(np.max(anchors.s)),
        int(dcfg["evaluation_s_points"]),
    )
    t_values = np.linspace(
        float(dcfg["transverse_min"]),
        float(dcfg["transverse_max"]),
        int(dcfg["evaluation_t_points"]),
    )
    ss, tt = np.meshgrid(s_values, t_values, indexing="ij")
    grid = np.column_stack([ss.ravel(), tt.ravel()])

    raw_chunks = []
    jacobian_chunks = []
    flatness_chunks = []
    chunk_size = 128
    for start in range(0, grid.shape[0], chunk_size):
        coordinates = torch.as_tensor(
            grid[start:start + chunk_size], dtype=torch.float64, device=device
        )
        raw, jacobian = output_and_jacobian(model, coordinates, create_graph=False)
        coefficients = real_output_to_complex(raw)
        flatness = discovery.component_normalized_loss(
            coefficients,
            evaluation_data,
            float(scfg["denominator_epsilon"]),
        )
        raw_chunks.append(raw.detach().cpu().numpy())
        jacobian_chunks.append(jacobian.detach().cpu().numpy())
        flatness_chunks.append(flatness.detach().cpu().numpy())

    raw_np = np.concatenate(raw_chunks, axis=0)
    jacobian_np = np.concatenate(jacobian_chunks, axis=0)
    flatness_np = np.concatenate(flatness_chunks, axis=0)
    coefficients_np = raw_np[:, :4] + 1j * raw_np[:, 4:]

    singular_values_np = np.linalg.svd(jacobian_np, compute_uv=False)
    sigma_ratio = singular_values_np[:, -1] / np.maximum(singular_values_np[:, 0], 1.0e-300)
    tangent_s = jacobian_np[..., 0]
    tangent_t = jacobian_np[..., 1]
    g_ss = np.sum(tangent_s * tangent_s, axis=-1)
    g_tt = np.sum(tangent_t * tangent_t, axis=-1)
    g_st = np.sum(tangent_s * tangent_t, axis=-1)
    sine_squared = np.clip(
        (g_ss * g_tt - g_st * g_st) / np.maximum(g_ss * g_tt, 1.0e-300),
        0.0,
        1.0,
    )

    a = coefficients_np[:, 0]
    b = coefficients_np[:, 1]
    c = coefficients_np[:, 2]
    d = coefficients_np[:, 3]
    abs_a_minus_one = np.abs(a - 1.0)
    abs_c_minus_one = np.abs(c - 1.0)
    abs_bd_minus_one = np.abs(b * d - 1.0)

    anchor_coordinates = torch.as_tensor(
        np.column_stack([anchors.s, np.zeros_like(anchors.s)]),
        dtype=torch.float64,
        device=device,
    )
    with torch.no_grad():
        anchor_prediction = model(anchor_coordinates).cpu().numpy()
    anchor_target = complex_coefficients_to_real(anchors.coefficients)
    anchor_rmse = float(np.sqrt(np.mean((anchor_prediction - anchor_target) ** 2)))

    coefficient_grid = coefficients_np.reshape(ss.shape + (4,))
    central_s_index = int(np.argmin(np.abs(s_values)))
    phase_line = np.unwrap(np.angle(coefficient_grid[central_s_index, :, 1]))
    phase_span = float(np.max(phase_line) - np.min(phase_line))

    summary = {
        "s_min": float(s_values[0]),
        "s_max": float(s_values[-1]),
        "t_min": float(t_values[0]),
        "t_max": float(t_values[-1]),
        "anchor_rmse": anchor_rmse,
        "flatness_median": float(np.median(flatness_np)),
        "flatness_max": float(np.max(flatness_np)),
        "jacobian_sigma_ratio_median": float(np.median(sigma_ratio)),
        "jacobian_sigma_ratio_min": float(np.min(sigma_ratio)),
        "fraction_sigma_ratio_below_0p05": float(np.mean(sigma_ratio < 0.05)),
        "sine_squared_median": float(np.median(sine_squared)),
        "sine_squared_min": float(np.min(sine_squared)),
        "max_abs_a_minus_one": float(np.max(abs_a_minus_one)),
        "max_abs_c_minus_one": float(np.max(abs_c_minus_one)),
        "max_abs_bd_minus_one": float(np.max(abs_bd_minus_one)),
        "median_abs_bd_minus_one": float(np.median(abs_bd_minus_one)),
        "central_line_b_phase_span": phase_span,
    }

    arrays = {
        "s_values": s_values,
        "t_values": t_values,
        "raw_output": raw_np.reshape(ss.shape + (8,)),
        "coefficients_real": coefficients_np.real.reshape(ss.shape + (4,)),
        "coefficients_imag": coefficients_np.imag.reshape(ss.shape + (4,)),
        "flatness": flatness_np.reshape(ss.shape),
        "jacobian_singular_values": singular_values_np.reshape(ss.shape + (2,)),
        "jacobian_sigma_ratio": sigma_ratio.reshape(ss.shape),
        "sine_squared": sine_squared.reshape(ss.shape),
        "abs_a_minus_one": abs_a_minus_one.reshape(ss.shape),
        "abs_c_minus_one": abs_c_minus_one.reshape(ss.shape),
        "abs_bd_minus_one": abs_bd_minus_one.reshape(ss.shape),
        "anchor_s": anchors.s,
        "anchor_coefficients_real": anchors.coefficients.real,
        "anchor_coefficients_imag": anchors.coefficients.imag,
        "anchor_prediction": anchor_prediction,
    }
    return summary, arrays


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg["seed"])

    anchors = load_continuation_trace(
        ROOT / cfg["input_trace"],
        int(cfg["anchors"]["n_points"]),
    )

    scfg = cfg["sampling"]
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

    all_summary: dict[str, dict] = {
        "chart_coordinate": {
            "definition": "z=s+i t; s is the signed pseudo-arclength coordinate on the Experiment 06 anchor path",
            "canonical_spectral_parameter_claimed": False,
        }
    }

    for variant_index, variant in enumerate(cfg["variants"]):
        name = str(variant["name"])
        model, history = train_variant(
            name=name,
            use_rank_penalty=bool(variant["use_rank_penalty"]),
            cfg=cfg,
            anchors=anchors,
            flatness_data=flatness_data,
            device=device,
            seed=seed + 1000 * variant_index,
        )
        summary, arrays = evaluate_variant(model, cfg, anchors, evaluation_data, device)
        summary["use_rank_penalty"] = bool(variant["use_rank_penalty"])
        all_summary[name] = summary

        (output_dir / f"history_{name}.json").write_text(json.dumps(history, indent=2))
        np.savez_compressed(output_dir / f"evaluation_{name}.npz", **arrays)
        torch.save(model.state_dict(), output_dir / f"model_{name}.pt")
        print(json.dumps({"variant": name, "evaluation": summary}, indent=2))

    (output_dir / "summary.json").write_text(json.dumps(all_summary, indent=2))
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
