from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "summary.json").read_text())
    data = np.load(args.results / "discovery_data.npz")

    final_losses = data["final_losses"]
    converged_mask = data["converged_mask"].astype(bool)
    converged = data["converged"]
    singular_values = data["singular_values"]
    invariant_matrix = data["invariant_matrix"]
    exponents = data["invariant_exponents"]
    product = data["product"]

    plt.figure(figsize=(7.0, 4.4))
    values = np.log10(np.maximum(final_losses, 1.0e-40))
    plt.hist(values, bins=32)
    plt.axvline(np.log10(summary["convergence_threshold"]), linestyle="--", label="Convergence threshold")
    plt.xlabel(r"$\log_{10}$ final component-normalized loss")
    plt.ylabel("Number of random initializations")
    plt.title("Landing on the flat-connection solution set")
    plt.legend()
    save(args.output / "landing_loss_histogram.png")

    x = np.arange(1, 9)
    plt.figure(figsize=(7.2, 4.8))
    for row in singular_values:
        plt.semilogy(x, row, alpha=0.08)
    plt.semilogy(x, np.median(singular_values, axis=0), linewidth=2.2, label="Median")
    plt.xticks(x)
    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value of column-normalized Jacobian")
    plt.title("Jacobian spectrum at converged flat connections")
    plt.legend()
    plt.grid(True, which="both", alpha=0.22)
    save(args.output / "jacobian_singular_values.png")

    plt.figure(figsize=(6.0, 5.2))
    plt.scatter(invariant_matrix[:, 0], -invariant_matrix[:, 1], s=8, alpha=0.35)
    lo = min(np.min(invariant_matrix[:, 0]), np.min(-invariant_matrix[:, 1]))
    hi = max(np.max(invariant_matrix[:, 0]), np.max(-invariant_matrix[:, 1]))
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Identity")
    plt.xlabel(r"$\operatorname{Re/Im}(\delta b/b)$")
    plt.ylabel(r"$-\operatorname{Re/Im}(\delta d/d)$")
    plt.title("Tangent relation inferred from Jacobian null vectors")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "null_tangent_invariant_relation.png")

    log_b = np.log(np.abs(converged[:, 1]))
    log_d = np.log(np.abs(converged[:, 3]))
    slope = summary["log_abs_d_vs_log_abs_b_slope"]
    intercept = float(np.mean(log_d - slope * log_b))
    xx = np.linspace(np.min(log_b), np.max(log_b), 200)
    plt.figure(figsize=(6.1, 5.0))
    plt.scatter(log_b, log_d, s=13, alpha=0.55)
    plt.plot(xx, slope * xx + intercept, linestyle="--", label=f"fit slope = {slope:.6f}")
    plt.xlabel(r"$\log |b|$")
    plt.ylabel(r"$\log |d|$")
    plt.title("Converged coefficient cloud")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "coefficient_cloud_log_modulus.png")

    arg_b = np.angle(converged[:, 1])
    arg_d = np.angle(converged[:, 3])
    phase_slope = summary["arg_d_vs_arg_b_slope"]
    phase_intercept = float(np.mean(arg_d - phase_slope * arg_b))
    xx = np.linspace(np.min(arg_b), np.max(arg_b), 200)
    plt.figure(figsize=(6.1, 5.0))
    plt.scatter(arg_b, arg_d, s=13, alpha=0.55)
    plt.plot(xx, phase_slope * xx + phase_intercept, linestyle="--", label=f"fit slope = {phase_slope:.6f}")
    plt.xlabel(r"$\arg b$")
    plt.ylabel(r"$\arg d$")
    plt.title("Phase relation in the converged coefficient cloud")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "coefficient_cloud_phase.png")

    plt.figure(figsize=(6.2, 5.0))
    plt.scatter(product.real, product.imag, s=13, alpha=0.55)
    plt.scatter([1.0], [0.0], marker="x", s=90, label="Exact value 1")
    plt.xlabel(r"$\operatorname{Re}(bd)$")
    plt.ylabel(r"$\operatorname{Im}(bd)$")
    plt.title("Product inferred from converged coefficients")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "product_bd.png")

    lines = [
        "# Experiment 05: S2 flat-family discovery from the Jacobian null space",
        "",
        "This is an independent reproduction of the discovery stage in the collaborator note. The relations a=c=1 and bd=1 are not imposed in the landing loss or in the Jacobian analysis.",
        "",
        f"- Random initializations: {summary['n_initializations']}",
        f"- Converged initializations: {summary['n_converged']}",
        f"- Landing batch size: {summary['landing_batch_size']}",
        f"- Jacobian diagnostic batch size: {summary['jacobian_batch_size']}",
        f"- Nullity histogram: {summary['nullity_histogram']}",
        f"- Median ratio s6/s7: {summary['median_ratio_s6_to_s7']:.6e}",
        f"- Maximum |delta a| in the two numerical null vectors: {summary['max_null_a_component']:.6e}",
        f"- Maximum |delta c| in the two numerical null vectors: {summary['max_null_c_component']:.6e}",
        f"- Median |a-1|: {summary['median_abs_a_minus_one']:.6e}",
        f"- Median |c-1|: {summary['median_abs_c_minus_one']:.6e}",
        f"- Invariant exponents normalized to p=1: ({exponents[0]:.6f}, {exponents[1]:.6f})",
        f"- Singular-value ratio in the invariant fit: {summary['invariant_singular_value_ratio']:.6e}",
        f"- Slope log|d| versus log|b|: {summary['log_abs_d_vs_log_abs_b_slope']:.6f}",
        f"- Slope arg(d) versus arg(b): {summary['arg_d_vs_arg_b_slope']:.6f}",
        f"- Maximum |bd-1|: {summary['max_abs_product_minus_one']:.6e}",
        f"- Sampled family coverage in |b|: {summary['b_abs_min']:.4f} to {summary['b_abs_max']:.4f}",
        "",
        "## Collaborator-note reference",
        "",
        "The collaborator note used 256 random initializations on a fixed batch of 16384 on-shell samples and retained 243 converged points. It reported nullity two at every retained point, a Jacobian singular-value gap of about 1e13--1e15, frozen a and c directions, invariant exponents (1,1), slopes -1 for both modulus and phase relations, and bd=1 to about 5e-6.",
        "",
        "The landing optimizer and its hyperparameters are not specified in the available note. This reproduction therefore uses a smaller landing batch and an explicitly documented Adam optimization, while keeping the Jacobian diagnostic batch at 16384 samples. Agreement should be judged by the recovered manifold dimension and algebraic structure, not by the exact number of converged initializations.",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
