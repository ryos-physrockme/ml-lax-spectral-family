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
DISCOVERY_TRAIN = ROOT / "experiments" / "05_s2_jacobian_family_discovery" / "train.py"
spec = importlib.util.spec_from_file_location("s2_discovery_train_for_continuation", DISCOVERY_TRAIN)
assert spec is not None and spec.loader is not None
discovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = discovery
spec.loader.exec_module(discovery)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def pair_to_complex(pair: list[float]) -> complex:
    return complex(float(pair[0]), float(pair[1]))


def complex_to_real_coordinates(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.complex128)
    return np.concatenate([point.real, point.imag])


def real_to_complex_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return coordinates[:4] + 1j * coordinates[4:]


def physical_null_basis(point: np.ndarray, jacobian_data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis of the two-dimensional real tangent space.

    The singular-value decomposition is performed after column normalization
    for conditioning. The null vectors are transformed back to the original
    coefficient coordinates before the basis is orthonormalized.
    """

    jac = discovery.realified_curvature_jacobian(point, jacobian_data)
    norms = np.linalg.norm(jac, axis=0)
    jac_normalized = jac / norms[None, :]
    _, singular_values, vh = np.linalg.svd(jac_normalized, full_matrices=False)
    tangents = vh[-2:, :] / norms[None, :]
    q, _ = np.linalg.qr(tangents.T)
    basis = q[:, :2].T
    return basis, singular_values


def radial_gradient(point: np.ndarray) -> np.ndarray:
    b = point[1]
    magnitude = abs(b)
    if magnitude == 0.0:
        raise ValueError("Cannot define a radial direction at b=0")
    gradient = np.zeros(8, dtype=np.float64)
    gradient[1] = b.real / magnitude
    gradient[5] = b.imag / magnitude
    return gradient


def project_to_null_space(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projected = basis.T @ (basis @ vector)
    norm = np.linalg.norm(projected)
    if norm < 1.0e-14:
        raise RuntimeError("Projection onto the Jacobian null space is numerically zero")
    return projected / norm


def correct_prediction(
    predicted_real: np.ndarray,
    corrector_data,
    epsilon: float,
    target_loss: float,
    learning_rate: float,
    steps: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float, int]:
    """Minimize flatness from a predictor and stop as soon as tolerance is met.

    Adam uses parameter-normalized steps even when the gradient is already very
    small. Running a fixed number of steps can therefore move an already
    acceptable predictor away from the flat manifold. We evaluate the loss
    before every optimizer step, return immediately once the requested
    tolerance is reached, and otherwise retain the best iterate seen.
    """

    raw_np = np.stack([predicted_real[:4], predicted_real[4:]], axis=-1)
    raw = torch.nn.Parameter(torch.as_tensor(raw_np, dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)

    best_loss = float("inf")
    best_raw = raw.detach().clone()
    best_step = 0

    for corrector_step in range(steps + 1):
        parameters = torch.complex(raw[:, 0], raw[:, 1])[None, :]
        loss = discovery.component_normalized_loss(parameters, corrector_data, epsilon)[0]
        loss_value = float(loss.detach().cpu())

        if loss_value < best_loss:
            best_loss = loss_value
            best_raw = raw.detach().clone()
            best_step = corrector_step

        if loss_value <= target_loss or corrector_step == steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    best_np = best_raw.cpu().numpy()
    corrected = np.concatenate([best_np[:, 0], best_np[:, 1]])
    correction_norm = float(np.linalg.norm(corrected - predicted_real))
    return corrected, best_loss, correction_norm, best_step


def trace_direction(
    start_point: np.ndarray,
    direction_sign: int,
    cfg: dict,
    corrector_data,
    jacobian_data: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict]:
    ccfg = cfg["continuation"]
    ocfg = cfg["corrector"]
    epsilon = float(ccfg["denominator_epsilon"])
    accepted_threshold = float(ccfg["accepted_loss_threshold"])
    h = float(ccfg["initial_step_length"])
    h_min = float(ccfg["minimum_step_length"])
    h_max = float(ccfg["maximum_step_length"])
    max_steps = int(ccfg["maximum_steps_each_direction"])
    max_retries = int(ccfg["maximum_retries_each_step"])
    target_min = float(ccfg["target_abs_b_min"])
    target_max = float(ccfg["target_abs_b_max"])

    current = complex_to_real_coordinates(start_point)
    basis, svals = physical_null_basis(start_point, jacobian_data)
    tangent = project_to_null_space(radial_gradient(start_point), basis)
    if direction_sign < 0:
        tangent = -tangent

    records: list[dict] = []
    signed_arclength = 0.0

    def make_record(
        step: int,
        point_real: np.ndarray,
        loss: float,
        step_length: float,
        correction: float,
        corrector_steps: int,
        singular_values: np.ndarray,
    ) -> dict:
        point = real_to_complex_coordinates(point_real)
        return {
            "direction": int(direction_sign),
            "step": int(step),
            "signed_arclength": float(signed_arclength),
            "step_length": float(step_length),
            "component_normalized_loss": float(loss),
            "correction_norm": float(correction),
            "corrector_steps": int(corrector_steps),
            "a": [float(point[0].real), float(point[0].imag)],
            "b": [float(point[1].real), float(point[1].imag)],
            "c": [float(point[2].real), float(point[2].imag)],
            "d": [float(point[3].real), float(point[3].imag)],
            "abs_b": float(abs(point[1])),
            "abs_d": float(abs(point[3])),
            "abs_bd_minus_one": float(abs(point[1] * point[3] - 1.0)),
            "jacobian_singular_values": [float(x) for x in singular_values],
        }

    start_tensor = torch.as_tensor(start_point[None, :], dtype=torch.complex128, device=device)
    start_loss = float(discovery.component_normalized_loss(start_tensor, corrector_data, epsilon)[0].detach().cpu())
    records.append(make_record(0, current, start_loss, 0.0, 0.0, 0, svals))

    for step in range(1, max_steps + 1):
        current_point = real_to_complex_coordinates(current)
        abs_b_now = abs(current_point[1])
        if direction_sign > 0 and abs_b_now >= target_max:
            break
        if direction_sign < 0 and abs_b_now <= target_min:
            break

        basis, _ = physical_null_basis(current_point, jacobian_data)
        tangent_new = project_to_null_space(tangent, basis)
        if np.dot(tangent_new, tangent) < 0.0:
            tangent_new = -tangent_new
        tangent = tangent_new

        accepted = False
        trial_h = h
        corrector_steps_used = 0
        for _ in range(max_retries):
            predicted = current + trial_h * tangent
            corrected, loss, correction_norm, corrector_steps_used = correct_prediction(
                predicted,
                corrector_data,
                epsilon,
                accepted_threshold,
                float(ocfg["learning_rate"]),
                int(ocfg["steps"]),
                device,
            )
            if np.isfinite(loss) and loss <= accepted_threshold:
                accepted = True
                break
            trial_h *= 0.5
            if trial_h < h_min:
                break

        if not accepted:
            print(
                f"direction={direction_sign:+d} stopped: corrector could not reach loss <= {accepted_threshold:.1e} "
                f"with step length >= {h_min:.3e}"
            )
            break

        current = corrected
        signed_arclength += direction_sign * trial_h
        h = trial_h
        if loss < accepted_threshold * 1.0e-2 and correction_norm < 0.35 * trial_h:
            h = min(h_max, h * 1.05)

        corrected_point = real_to_complex_coordinates(current)
        basis_after, svals_after = physical_null_basis(corrected_point, jacobian_data)
        tangent_after = project_to_null_space(tangent, basis_after)
        if np.dot(tangent_after, tangent) < 0.0:
            tangent_after = -tangent_after
        tangent = tangent_after
        records.append(
            make_record(
                step,
                current,
                loss,
                trial_h,
                correction_norm,
                corrector_steps_used,
                svals_after,
            )
        )

        if step % 50 == 0:
            print(
                f"direction={direction_sign:+d} step={step:4d} |b|={abs(corrected_point[1]):.6f} "
                f"|d|={abs(corrected_point[3]):.6f} loss={loss:.3e} h={trial_h:.3e} "
                f"corrector_steps={corrector_steps_used}"
            )

    return records


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg["seed"])

    seed_path = ROOT / cfg["input_seed"]
    seed_payload = json.loads(seed_path.read_text())
    start_point = np.asarray(
        [
            pair_to_complex(seed_payload["a"]),
            pair_to_complex(seed_payload["b"]),
            pair_to_complex(seed_payload["c"]),
            pair_to_complex(seed_payload["d"]),
        ],
        dtype=np.complex128,
    )

    scfg = cfg["sampling"]
    corrector_data = discovery.sample_onshell_data(
        int(scfg["corrector_samples"]),
        np.random.default_rng(seed),
        device,
        float(scfg["theta_margin"]),
    )
    jacobian_data_torch = discovery.sample_onshell_data(
        int(scfg["jacobian_samples"]),
        np.random.default_rng(seed + 1),
        torch.device("cpu"),
        float(scfg["theta_margin"]),
    )
    jacobian_data = discovery.data_as_numpy(jacobian_data_torch)

    outward = trace_direction(start_point, +1, cfg, corrector_data, jacobian_data, device)
    inward = trace_direction(start_point, -1, cfg, corrector_data, jacobian_data, device)

    combined = list(reversed(inward[1:])) + outward
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps(combined, indent=2))
    (out / "outward_trace.json").write_text(json.dumps(outward, indent=2))
    (out / "inward_trace.json").write_text(json.dumps(inward, indent=2))

    def cplx(row: dict, key: str) -> complex:
        return complex(*row[key])

    b_values = np.asarray([cplx(row, "b") for row in combined])
    d_values = np.asarray([cplx(row, "d") for row in combined])
    a_values = np.asarray([cplx(row, "a") for row in combined])
    c_values = np.asarray([cplx(row, "c") for row in combined])
    losses = np.asarray([row["component_normalized_loss"] for row in combined])
    corrector_steps = np.asarray([row["corrector_steps"] for row in combined])

    summary = {
        "seed_provenance": seed_payload["provenance"],
        "n_points_total": len(combined),
        "n_points_outward_including_start": len(outward),
        "n_points_inward_including_start": len(inward),
        "abs_b_min": float(np.min(np.abs(b_values))),
        "abs_b_max": float(np.max(np.abs(b_values))),
        "abs_d_min": float(np.min(np.abs(d_values))),
        "abs_d_max": float(np.max(np.abs(d_values))),
        "max_component_normalized_loss": float(np.max(losses)),
        "max_abs_a_minus_one": float(np.max(np.abs(a_values - 1.0))),
        "max_abs_c_minus_one": float(np.max(np.abs(c_values - 1.0))),
        "max_abs_bd_minus_one": float(np.max(np.abs(b_values * d_values - 1.0))),
        "max_corrector_steps_used": int(np.max(corrector_steps)),
        "median_corrector_steps_used": float(np.median(corrector_steps)),
        "minimum_jacobian_s6_over_s7": float(
            min(
                row["jacobian_singular_values"][5]
                / max(row["jacobian_singular_values"][6], 1.0e-300)
                for row in combined
            )
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
