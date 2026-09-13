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


def complex_column(trace: list[dict], key: str) -> np.ndarray:
    return np.asarray([complex(*row[key]) for row in trace])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = json.loads((args.results / "trace.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())

    s = np.asarray([row["signed_arclength"] for row in trace])
    a = complex_column(trace, "a")
    b = complex_column(trace, "b")
    c = complex_column(trace, "c")
    d = complex_column(trace, "d")
    loss = np.asarray([row["component_normalized_loss"] for row in trace])

    plt.figure(figsize=(7.0, 5.2))
    scatter = plt.scatter(b.real, b.imag, c=s, s=14)
    plt.colorbar(scatter, label="Signed continuation arclength")
    plt.scatter([b[np.argmin(np.abs(s))].real], [b[np.argmin(np.abs(s))].imag], marker="*", s=120, label="Learned seed")
    plt.xlabel(r"$\operatorname{Re} b$")
    plt.ylabel(r"$\operatorname{Im} b$")
    plt.title(r"Continuation path in the complex $b$ plane")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "continuation_b_plane.png")

    plt.figure(figsize=(7.0, 5.2))
    scatter = plt.scatter(d.real, d.imag, c=s, s=14)
    plt.colorbar(scatter, label="Signed continuation arclength")
    plt.xlabel(r"$\operatorname{Re} d$")
    plt.ylabel(r"$\operatorname{Im} d$")
    plt.title(r"Continuation path in the complex $d$ plane")
    plt.grid(True, alpha=0.25)
    save(args.output / "continuation_d_plane.png")

    plt.figure(figsize=(7.4, 4.8))
    plt.semilogy(s, np.abs(b), label=r"$|b|$")
    plt.semilogy(s, np.abs(d), label=r"$|d|$")
    plt.xlabel("Signed continuation arclength")
    plt.ylabel("Coefficient magnitude")
    plt.title("Opposite motion of the two varying coefficients")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    save(args.output / "coefficient_magnitudes_vs_arclength.png")

    plt.figure(figsize=(7.4, 4.8))
    plt.semilogy(s, np.maximum(np.abs(a - 1.0), 1.0e-18), label=r"$|a-1|$")
    plt.semilogy(s, np.maximum(np.abs(c - 1.0), 1.0e-18), label=r"$|c-1|$")
    plt.semilogy(s, np.maximum(np.abs(b * d - 1.0), 1.0e-18), label=r"$|bd-1|$")
    plt.xlabel("Signed continuation arclength")
    plt.ylabel("Independent analytic check")
    plt.title("Relations inferred in Experiment 05 along the continuation path")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    save(args.output / "family_relations_vs_arclength.png")

    plt.figure(figsize=(7.4, 4.8))
    plt.semilogy(s, np.maximum(loss, 1.0e-20))
    plt.xlabel("Signed continuation arclength")
    plt.ylabel("Component-normalized flatness residual")
    plt.title("Flatness accuracy along the continuation path")
    plt.grid(True, which="both", alpha=0.25)
    save(args.output / "flatness_vs_arclength.png")

    lines = [
        "# Experiment 06: one-dimensional continuation through the S2 Lax family",
        "",
        "The starting point is a converged coefficient vector produced by Experiment 05. The continuation code never substitutes the analytic equations a=c=1 or bd=1 into the predictor, tangent calculation, or corrector.",
        "",
        f"- Total corrected points: {summary['n_points_total']}",
        f"- Points toward increasing |b|, including the start: {summary['n_points_outward_including_start']}",
        f"- Points toward decreasing |b|, including the start: {summary['n_points_inward_including_start']}",
        f"- Coverage in |b|: {summary['abs_b_min']:.6f} to {summary['abs_b_max']:.6f}",
        f"- Coverage in |d|: {summary['abs_d_min']:.6f} to {summary['abs_d_max']:.6f}",
        f"- Maximum component-normalized flatness residual: {summary['max_component_normalized_loss']:.6e}",
        f"- Maximum |a-1|: {summary['max_abs_a_minus_one']:.6e}",
        f"- Maximum |c-1|: {summary['max_abs_c_minus_one']:.6e}",
        f"- Maximum |bd-1|: {summary['max_abs_bd_minus_one']:.6e}",
        f"- Minimum Jacobian spectral gap s6/s7 along the stored path: {summary['minimum_jacobian_s6_over_s7']:.6e}",
        "",
        "## Interpretation",
        "",
        "The signed arclength is a reproducible coordinate on this particular one-dimensional path through the two-real-dimensional flat-connection solution manifold. It should not be identified with a unique or canonical complex spectral parameter. Different initial tangent choices can trace different one-dimensional curves in the same complex-one-dimensional family.",
        "",
        "## Collaborator-note reference",
        "",
        "The collaborator note reports a predictor-corrector path with about 1200 accepted steps, coverage |b| approximately 0.05 to 18.7, component-normalized residual at or below 1e-10, frozen coefficients stable to about 1.5e-5, and |bd-1| at or below about 6.6e-5. It also reports a numerical fold near the large-|b| end caused by tangential drift of the Adam corrector. The present validation deliberately stops at |b|=12 so that the clean monotonic region can be tested before studying that fold separately.",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
