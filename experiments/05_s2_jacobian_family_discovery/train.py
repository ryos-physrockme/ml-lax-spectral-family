from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class OnShellData:
    sin_theta: torch.Tensor
    cos_theta: torch.Tensor
    phi_plus: torch.Tensor
    phi_minus: torch.Tensor
    theta_plus: torch.Tensor
    theta_minus: torch.Tensor
    phi_plus_minus: torch.Tensor
    theta_plus_minus: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sample_onshell_data(
    n: int,
    rng: np.random.Generator,
    device: torch.device,
    theta_margin: float,
) -> OnShellData:
    theta = rng.uniform(theta_margin, np.pi - theta_margin, size=n)
    phi_plus = rng.normal(size=n)
    phi_minus = rng.normal(size=n)
    theta_plus = rng.normal(size=n)
    theta_minus = rng.normal(size=n)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_plus_minus = sin_theta * cos_theta * phi_plus * phi_minus
    phi_plus_minus = -(
        cos_theta
        * (theta_plus * phi_minus + theta_minus * phi_plus)
        / sin_theta
    )

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float64, device=device)

    return OnShellData(
        sin_theta=tensor(sin_theta),
        cos_theta=tensor(cos_theta),
        phi_plus=tensor(phi_plus),
        phi_minus=tensor(phi_minus),
        theta_plus=tensor(theta_plus),
        theta_minus=tensor(theta_minus),
        phi_plus_minus=tensor(phi_plus_minus),
        theta_plus_minus=tensor(theta_plus_minus),
    )


def connection_terms(parameters: torch.Tensor, data: OnShellData) -> tuple[torch.Tensor, ...]:
    """Return d_+ L_-, d_- L_+, [L_+,L_-], and curvature.

    parameters has shape (n_initializations, 4) and is ordered as (a,b,c,d).
    Lie-algebra components are ordered as (T1,T2,T3).
    """

    a, b, c, d = [parameters[:, i] for i in range(4)]
    s = data.sin_theta
    ct = data.cos_theta
    pp = data.phi_plus
    pm = data.phi_minus
    tp = data.theta_plus
    tm = data.theta_minus
    ppm = data.phi_plus_minus
    tpm = data.theta_plus_minus

    d_plus_l_minus = torch.stack(
        [
            d[:, None] * (ct * tp * pm + s * ppm),
            -d[:, None] * tpm,
            c[:, None] * (-s * tp * pm + ct * ppm),
        ],
        dim=-1,
    )
    d_minus_l_plus = torch.stack(
        [
            b[:, None] * (ct * tm * pp + s * ppm),
            -b[:, None] * tpm,
            a[:, None] * (-s * tm * pp + ct * ppm),
        ],
        dim=-1,
    )

    l_plus = torch.stack(
        [
            b[:, None] * s * pp,
            -b[:, None] * tp,
            a[:, None] * ct * pp,
        ],
        dim=-1,
    )
    l_minus = torch.stack(
        [
            d[:, None] * s * pm,
            -d[:, None] * tm,
            c[:, None] * ct * pm,
        ],
        dim=-1,
    )
    commutator = torch.cross(l_plus, l_minus, dim=-1)
    curvature = d_plus_l_minus - d_minus_l_plus + commutator
    return d_plus_l_minus, d_minus_l_plus, commutator, curvature


def component_normalized_loss(
    parameters: torch.Tensor,
    data: OnShellData,
    epsilon: float,
) -> torch.Tensor:
    term_1, term_2, term_3, curvature = connection_terms(parameters, data)

    def squared_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(x) ** 2, dim=-1)

    numerator = squared_norm(curvature)
    denominator = (
        squared_norm(term_1)
        + squared_norm(term_2)
        + squared_norm(term_3)
        + epsilon
    )
    return torch.mean(numerator / denominator, dim=1)


def data_as_numpy(data: OnShellData) -> dict[str, np.ndarray]:
    return {
        name: getattr(data, name).detach().cpu().numpy()
        for name in data.__dataclass_fields__
    }


def realified_curvature_jacobian(
    parameters: np.ndarray,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    """Analytic real Jacobian of the complex curvature residual.

    The curvature is holomorphic in the four complex coefficients (a,b,c,d).
    We first construct dF/d(a,b,c,d), then realify it using
    dF/d Im(z) = i dF/dz.  The output coordinate ordering is
    (Re F, Im F); the parameter ordering is
    (Re a,Re b,Re c,Re d,Im a,Im b,Im c,Im d).
    """

    a, b, c, d = parameters
    s = data["sin_theta"]
    ct = data["cos_theta"]
    pp = data["phi_plus"]
    pm = data["phi_minus"]
    tp = data["theta_plus"]
    tm = data["theta_minus"]
    ppm = data["phi_plus_minus"]
    tpm = data["theta_plus_minus"]
    x = s * ct * pp * pm

    n = s.size
    jac_complex = np.empty((n, 3, 4), dtype=np.complex128)

    jac_complex[:, 0, 0] = ct * d * pp * tm
    jac_complex[:, 0, 1] = -s * ppm + ct * (-c * pm * tp - pp * tm)
    jac_complex[:, 0, 2] = -ct * b * pm * tp
    jac_complex[:, 0, 3] = s * ppm + ct * (pm * tp + a * pp * tm)

    jac_complex[:, 1, 0] = d * x
    jac_complex[:, 1, 1] = tpm - c * x
    jac_complex[:, 1, 2] = -b * x
    jac_complex[:, 1, 3] = -tpm + a * x

    jac_complex[:, 2, 0] = -ct * ppm + s * pp * tm
    jac_complex[:, 2, 1] = s * d * (pm * tp - pp * tm)
    jac_complex[:, 2, 2] = ct * ppm - s * pm * tp
    jac_complex[:, 2, 3] = s * b * (pm * tp - pp * tm)

    j = jac_complex.reshape(-1, 4)
    real_columns = np.vstack([j.real, j.imag])
    imag_columns = np.vstack([-j.imag, j.real])
    return np.hstack([real_columns, imag_columns])


def analyze_jacobians(
    points: np.ndarray,
    data: dict[str, np.ndarray],
    small_threshold: float,
) -> dict[str, np.ndarray | float | int | list[float]]:
    singular_values = []
    null_vectors = []
    nullities = []
    column_norms = []

    for point in points:
        jac = realified_curvature_jacobian(point, data)
        norms = np.linalg.norm(jac, axis=0)
        jac_normalized = jac / norms[None, :]
        _, svals, vh = np.linalg.svd(jac_normalized, full_matrices=False)
        singular_values.append(svals)
        null_vectors.append(vh[-2:, :])
        nullities.append(int(np.count_nonzero(svals < small_threshold)))
        column_norms.append(norms)

    singular_values_arr = np.asarray(singular_values)
    null_vectors_arr = np.asarray(null_vectors)
    column_norms_arr = np.asarray(column_norms)

    # Convert each real tangent vector to complex coefficient variations.
    delta_complex = (
        null_vectors_arr[..., :4]
        + 1j * null_vectors_arr[..., 4:]
    )
    frozen_a = np.abs(delta_complex[..., 0])
    frozen_c = np.abs(delta_complex[..., 2])

    rows = []
    for point, two_tangents in zip(points, delta_complex):
        b = point[1]
        d = point[3]
        for tangent in two_tangents:
            delta_log_b = tangent[1] / b
            delta_log_d = tangent[3] / d
            rows.append([delta_log_b.real, delta_log_d.real])
            rows.append([delta_log_b.imag, delta_log_d.imag])
    invariant_matrix = np.asarray(rows, dtype=np.float64)
    _, invariant_singular_values, invariant_vh = np.linalg.svd(
        invariant_matrix, full_matrices=False
    )
    exponents = invariant_vh[-1]
    if exponents[0] < 0:
        exponents = -exponents
    exponents = exponents / exponents[0]

    log_b = np.log(np.abs(points[:, 1]))
    log_d = np.log(np.abs(points[:, 3]))
    log_slope, log_intercept = np.polyfit(log_b, log_d, 1)

    arg_b = np.angle(points[:, 1])
    arg_d = np.angle(points[:, 3])
    phase_slope, phase_intercept = np.polyfit(arg_b, arg_d, 1)

    product = points[:, 1] * points[:, 3]

    return {
        "singular_values": singular_values_arr,
        "null_vectors": null_vectors_arr,
        "column_norms": column_norms_arr,
        "nullities": np.asarray(nullities, dtype=np.int64),
        "frozen_a": frozen_a,
        "frozen_c": frozen_c,
        "invariant_matrix": invariant_matrix,
        "invariant_singular_values": invariant_singular_values,
        "invariant_exponents": exponents,
        "log_slope": float(log_slope),
        "log_intercept": float(log_intercept),
        "phase_slope": float(phase_slope),
        "phase_intercept": float(phase_intercept),
        "product": product,
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    lcfg = cfg["landing"]
    n_initializations = int(lcfg["n_initializations"])
    landing_data = sample_onshell_data(
        int(lcfg["n_samples"]),
        rng,
        device,
        float(lcfg["theta_margin"]),
    )

    raw = torch.nn.Parameter(
        float(lcfg["initialization_scale"])
        * torch.randn(n_initializations, 4, 2, dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([raw], lr=float(lcfg["learning_rate"]))
    history = []

    for step in range(1, int(lcfg["steps"]) + 1):
        parameters = torch.complex(raw[..., 0], raw[..., 1])
        losses = component_normalized_loss(
            parameters,
            landing_data,
            float(lcfg["denominator_epsilon"]),
        )
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()

        if step == 1 or step % int(lcfg["log_every"]) == 0 or step == int(lcfg["steps"]):
            detached = losses.detach().cpu().numpy()
            record = {
                "step": step,
                "loss_min": float(np.min(detached)),
                "loss_median": float(np.median(detached)),
                "loss_max": float(np.max(detached)),
            }
            history.append(record)
            print(json.dumps(record))

    parameters = torch.complex(raw[..., 0], raw[..., 1]).detach().cpu().numpy()
    final_losses = component_normalized_loss(
        torch.as_tensor(parameters, dtype=torch.complex128, device=device),
        landing_data,
        float(lcfg["denominator_epsilon"]),
    ).detach().cpu().numpy()

    threshold = float(lcfg["convergence_threshold"])
    converged_mask = final_losses < threshold
    converged = parameters[converged_mask]
    if converged.shape[0] == 0:
        raise RuntimeError("No random initialization reached the convergence threshold")

    jcfg = cfg["jacobian"]
    jacobian_data_torch = sample_onshell_data(
        int(jcfg["n_samples"]),
        np.random.default_rng(seed + 1),
        torch.device("cpu"),
        float(jcfg["theta_margin"]),
    )
    jacobian_data = data_as_numpy(jacobian_data_torch)
    analysis = analyze_jacobians(
        converged,
        jacobian_data,
        float(jcfg["small_singular_value_threshold"]),
    )

    product = np.asarray(analysis["product"])
    invariant_svals = np.asarray(analysis["invariant_singular_values"])
    exponents = np.asarray(analysis["invariant_exponents"])
    nullities = np.asarray(analysis["nullities"])
    frozen_a = np.asarray(analysis["frozen_a"])
    frozen_c = np.asarray(analysis["frozen_c"])
    svals = np.asarray(analysis["singular_values"])

    unique_nullity, counts = np.unique(nullities, return_counts=True)
    nullity_histogram = {str(int(k)): int(v) for k, v in zip(unique_nullity, counts)}

    summary = {
        "n_initializations": n_initializations,
        "n_converged": int(converged.shape[0]),
        "landing_batch_size": int(lcfg["n_samples"]),
        "jacobian_batch_size": int(jcfg["n_samples"]),
        "convergence_threshold": threshold,
        "final_loss_median_converged": float(np.median(final_losses[converged_mask])),
        "nullity_histogram": nullity_histogram,
        "median_singular_values": np.median(svals, axis=0).tolist(),
        "median_ratio_s6_to_s7": float(np.median(svals[:, 5] / np.maximum(svals[:, 6], 1.0e-300))),
        "max_null_a_component": float(np.max(frozen_a)),
        "max_null_c_component": float(np.max(frozen_c)),
        "median_abs_a_minus_one": float(np.median(np.abs(converged[:, 0] - 1.0))),
        "median_abs_c_minus_one": float(np.median(np.abs(converged[:, 2] - 1.0))),
        "max_abs_a_minus_one": float(np.max(np.abs(converged[:, 0] - 1.0))),
        "max_abs_c_minus_one": float(np.max(np.abs(converged[:, 2] - 1.0))),
        "invariant_exponents_normalized": exponents.tolist(),
        "invariant_singular_value_ratio": float(invariant_svals[0] / invariant_svals[-1]),
        "log_abs_d_vs_log_abs_b_slope": float(analysis["log_slope"]),
        "arg_d_vs_arg_b_slope": float(analysis["phase_slope"]),
        "mean_product_real": float(np.mean(product.real)),
        "mean_product_imag": float(np.mean(product.imag)),
        "max_abs_product_minus_one": float(np.max(np.abs(product - 1.0))),
        "b_abs_min": float(np.min(np.abs(converged[:, 1]))),
        "b_abs_max": float(np.max(np.abs(converged[:, 1]))),
    }

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(
        out / "discovery_data.npz",
        parameters=parameters,
        final_losses=final_losses,
        converged_mask=converged_mask,
        converged=converged,
        singular_values=svals,
        null_vectors=np.asarray(analysis["null_vectors"]),
        nullities=nullities,
        frozen_a=frozen_a,
        frozen_c=frozen_c,
        invariant_matrix=np.asarray(analysis["invariant_matrix"]),
        invariant_singular_values=invariant_svals,
        invariant_exponents=exponents,
        product=product,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
