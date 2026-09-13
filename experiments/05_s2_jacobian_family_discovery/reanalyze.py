from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.py"
spec = importlib.util.spec_from_file_location("s2_family_discovery_train", TRAIN_PATH)
assert spec is not None and spec.loader is not None
train_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_module
spec.loader.exec_module(train_module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def corrected_analysis(
    points: np.ndarray,
    data: dict[str, np.ndarray],
    small_threshold: float,
) -> dict[str, np.ndarray | float]:
    singular_values = []
    null_vectors_original_coordinates = []
    nullities = []
    column_norms = []

    for point in points:
        jac = train_module.realified_curvature_jacobian(point, data)
        norms = np.linalg.norm(jac, axis=0)
        jac_normalized = jac / norms[None, :]
        _, svals, vh = np.linalg.svd(jac_normalized, full_matrices=False)

        # If J_norm = J diag(1/n_i), a null vector v of J_norm corresponds
        # to the physical parameter-space tangent diag(1/n_i) v of J.
        tangents = vh[-2:, :] / norms[None, :]
        tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

        singular_values.append(svals)
        null_vectors_original_coordinates.append(tangents)
        nullities.append(int(np.count_nonzero(svals < small_threshold)))
        column_norms.append(norms)

    singular_values = np.asarray(singular_values)
    tangents_real = np.asarray(null_vectors_original_coordinates)
    nullities = np.asarray(nullities, dtype=np.int64)
    column_norms = np.asarray(column_norms)

    tangents_complex = tangents_real[..., :4] + 1j * tangents_real[..., 4:]
    frozen_a = np.abs(tangents_complex[..., 0])
    frozen_c = np.abs(tangents_complex[..., 2])

    usable = nullities == 2
    if not np.any(usable):
        raise RuntimeError("No converged point has the expected two-dimensional real null space")

    invariant_rows = []
    for point, two_tangents in zip(points[usable], tangents_complex[usable]):
        b = point[1]
        d = point[3]
        for tangent in two_tangents:
            delta_log_b = tangent[1] / b
            delta_log_d = tangent[3] / d
            invariant_rows.append([delta_log_b.real, delta_log_d.real])
            invariant_rows.append([delta_log_b.imag, delta_log_d.imag])
    invariant_matrix = np.asarray(invariant_rows, dtype=np.float64)
    _, invariant_singular_values, invariant_vh = np.linalg.svd(
        invariant_matrix, full_matrices=False
    )
    exponents = invariant_vh[-1]
    if exponents[0] < 0:
        exponents = -exponents
    exponents = exponents / exponents[0]

    points_for_cloud = points[usable]
    log_b = np.log(np.abs(points_for_cloud[:, 1]))
    log_d = np.log(np.abs(points_for_cloud[:, 3]))
    log_slope, log_intercept = np.polyfit(log_b, log_d, 1)
    arg_b = np.angle(points_for_cloud[:, 1])
    arg_d = np.angle(points_for_cloud[:, 3])
    phase_slope, phase_intercept = np.polyfit(arg_b, arg_d, 1)
    product = points_for_cloud[:, 1] * points_for_cloud[:, 3]

    return {
        "singular_values": singular_values,
        "null_vectors": tangents_real,
        "column_norms": column_norms,
        "nullities": nullities,
        "frozen_a": frozen_a,
        "frozen_c": frozen_c,
        "invariant_matrix": invariant_matrix,
        "invariant_singular_values": invariant_singular_values,
        "invariant_exponents": exponents,
        "usable_mask": usable,
        "points_for_cloud": points_for_cloud,
        "log_slope": float(log_slope),
        "log_intercept": float(log_intercept),
        "phase_slope": float(phase_slope),
        "phase_intercept": float(phase_intercept),
        "product": product,
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    result_file = args.results / "discovery_data.npz"
    old = np.load(result_file)
    parameters = old["parameters"]
    final_losses = old["final_losses"]
    converged_mask = old["converged_mask"].astype(bool)
    converged = old["converged"]

    jcfg = cfg["jacobian"]
    seed = int(cfg["seed"])
    jacobian_data_torch = train_module.sample_onshell_data(
        int(jcfg["n_samples"]),
        np.random.default_rng(seed + 1),
        torch.device("cpu"),
        float(jcfg["theta_margin"]),
    )
    jacobian_data = train_module.data_as_numpy(jacobian_data_torch)
    analysis = corrected_analysis(
        converged,
        jacobian_data,
        float(jcfg["small_singular_value_threshold"]),
    )

    svals = np.asarray(analysis["singular_values"])
    nullities = np.asarray(analysis["nullities"])
    frozen_a = np.asarray(analysis["frozen_a"])
    frozen_c = np.asarray(analysis["frozen_c"])
    invariant_svals = np.asarray(analysis["invariant_singular_values"])
    exponents = np.asarray(analysis["invariant_exponents"])
    usable = np.asarray(analysis["usable_mask"])
    cloud = np.asarray(analysis["points_for_cloud"])
    product = np.asarray(analysis["product"])

    unique_nullity, counts = np.unique(nullities, return_counts=True)
    nullity_histogram = {str(int(k)): int(v) for k, v in zip(unique_nullity, counts)}

    lcfg = cfg["landing"]
    summary = {
        "n_initializations": int(lcfg["n_initializations"]),
        "n_converged": int(converged.shape[0]),
        "n_points_used_for_tangent_invariant": int(np.count_nonzero(usable)),
        "landing_batch_size": int(lcfg["n_samples"]),
        "jacobian_batch_size": int(jcfg["n_samples"]),
        "convergence_threshold": float(lcfg["convergence_threshold"]),
        "final_loss_median_converged": float(np.median(final_losses[converged_mask])),
        "nullity_histogram": nullity_histogram,
        "median_singular_values": np.median(svals, axis=0).tolist(),
        "median_ratio_s6_to_s7": float(np.median(svals[:, 5] / np.maximum(svals[:, 6], 1.0e-300))),
        "max_null_a_component": float(np.max(frozen_a[usable])),
        "max_null_c_component": float(np.max(frozen_c[usable])),
        "median_abs_a_minus_one": float(np.median(np.abs(cloud[:, 0] - 1.0))),
        "median_abs_c_minus_one": float(np.median(np.abs(cloud[:, 2] - 1.0))),
        "max_abs_a_minus_one": float(np.max(np.abs(cloud[:, 0] - 1.0))),
        "max_abs_c_minus_one": float(np.max(np.abs(cloud[:, 2] - 1.0))),
        "invariant_exponents_normalized": exponents.tolist(),
        "invariant_singular_value_ratio": float(invariant_svals[0] / invariant_svals[-1]),
        "log_abs_d_vs_log_abs_b_slope": float(analysis["log_slope"]),
        "arg_d_vs_arg_b_slope": float(analysis["phase_slope"]),
        "mean_product_real": float(np.mean(product.real)),
        "mean_product_imag": float(np.mean(product.imag)),
        "max_abs_product_minus_one": float(np.max(np.abs(product - 1.0))),
        "b_abs_min": float(np.min(np.abs(cloud[:, 1]))),
        "b_abs_max": float(np.max(np.abs(cloud[:, 1]))),
        "jacobian_column_normalization_note": (
            "SVD is performed after unit-normalizing Jacobian columns; null vectors are divided by the corresponding column norms before interpreting coefficient-space tangents."
        ),
    }

    (args.results / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        result_file,
        parameters=parameters,
        final_losses=final_losses,
        converged_mask=converged_mask,
        converged=converged,
        singular_values=svals,
        null_vectors=np.asarray(analysis["null_vectors"]),
        column_norms=np.asarray(analysis["column_norms"]),
        nullities=nullities,
        frozen_a=frozen_a,
        frozen_c=frozen_c,
        usable_mask=usable,
        points_for_cloud=cloud,
        invariant_matrix=np.asarray(analysis["invariant_matrix"]),
        invariant_singular_values=invariant_svals,
        invariant_exponents=exponents,
        product=product,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
