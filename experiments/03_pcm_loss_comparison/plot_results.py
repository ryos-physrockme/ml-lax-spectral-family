from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("plain", "Bare flatness loss"),
    ("analytic_weight", "Analytic per-lambda weight"),
    ("normalized_adaptive", "Component normalization + adaptive weight"),
]
REFERENCE_10000 = {
    "plain": {"c_relative_l2_error": 1.7e-2, "spectral_curve_gap_mean": 5.9e-3, "spectral_curve_gap_max": 7.0e-2},
    "analytic_weight": {"c_relative_l2_error": 7.6e-3, "spectral_curve_gap_mean": 5.9e-3, "spectral_curve_gap_max": 3.0e-2},
    "normalized_adaptive": {"c_relative_l2_error": 1.1e-2, "spectral_curve_gap_mean": 9.7e-3, "spectral_curve_gap_max": 5.5e-2},
}


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


def load_metrics(root: Path, method: str, step: int) -> dict:
    return json.loads((root / method / f"metrics_step_{step}.json").read_text())


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels = [label for _, label in METHODS]
    metrics_10000 = {method: load_metrics(args.results, method, 10000) for method, _ in METHODS}
    metrics_3000 = {method: load_metrics(args.results, method, 3000) for method, _ in METHODS}

    plt.figure(figsize=(7.2, 4.4))
    for method, label in METHODS:
        history = json.loads((args.results / method / "history.json").read_text())
        steps = [row["step"] for row in history]
        losses = [row["training_loss"] for row in history]
        plt.semilogy(steps, losses, label=label)
    plt.xlabel("Training step")
    plt.ylabel("Training objective")
    plt.title("Principal chiral model: optimization histories")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save(args.output / "training_objectives.png")

    x = np.arange(len(METHODS))
    ours = [metrics_10000[m]["c_relative_l2_error"] for m, _ in METHODS]
    reported = [REFERENCE_10000[m]["c_relative_l2_error"] for m, _ in METHODS]
    width = 0.36
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, ours, width, label="Independent implementation")
    plt.bar(x + width / 2, reported, width, label="Collaborator note")
    plt.xticks(x, ["Bare", "Analytic weight", "Normalized + adaptive"])
    plt.ylabel(r"Relative $L^2$ error of $c(\lambda)$")
    plt.title("Hard parametrization, 10000 training steps")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save(args.output / "relative_l2_comparison.png")

    ours_gap = [metrics_10000[m]["spectral_curve_gap_max"] for m, _ in METHODS]
    reported_gap = [REFERENCE_10000[m]["spectral_curve_gap_max"] for m, _ in METHODS]
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, ours_gap, width, label="Independent implementation")
    plt.bar(x + width / 2, reported_gap, width, label="Collaborator note")
    plt.xticks(x, ["Bare", "Analytic weight", "Normalized + adaptive"])
    plt.ylabel(r"Maximum $|a+c-2ac|$")
    plt.title("Worst spectral-curve error after 10000 steps")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    save(args.output / "gap_max_comparison.png")

    plt.figure(figsize=(7.2, 4.5))
    data0 = np.load(args.results / METHODS[0][0] / "grid_step_10000.npz")
    lam_real = data0["lam_real"].real
    plt.plot(lam_real, data0["c_real_exact"].real, linestyle="--", label="Analytic target")
    for method, label in METHODS:
        data = np.load(args.results / method / "grid_step_10000.npz")
        plt.plot(data["lam_real"].real, data["c_real"].real, label=label)
    plt.xlabel(r"Real spectral parameter $\lambda$")
    plt.ylabel(r"$\operatorname{Re} c(\lambda)$")
    plt.title("Learned coefficient on the real spectral-parameter axis")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save(args.output / "real_axis_coefficients.png")

    for method, label in METHODS:
        data = np.load(args.results / method / "grid_step_10000.npz")
        lam = data["lam"]
        c_pred = data["c_pred"]
        c_exact = data["c_exact"]
        n = int(data["grid_size"])
        re = np.unique(lam.real)
        im = np.unique(lam.imag)
        error = np.abs(c_pred - c_exact).reshape(n, n)
        plt.figure(figsize=(6.0, 5.0))
        image = plt.imshow(
            np.log10(np.maximum(error, 1.0e-16)),
            origin="lower",
            extent=[re.min(), re.max(), im.min(), im.max()],
            aspect="auto",
        )
        plt.colorbar(image, label=r"$\log_{10}|c_{\rm learned}-c_{\rm exact}|$")
        plt.xlabel(r"$\operatorname{Re}\lambda$")
        plt.ylabel(r"$\operatorname{Im}\lambda$")
        plt.title(label)
        save(args.output / f"error_plane_{method}.png")

    lines = [
        "# Experiment 03: principal chiral model loss comparison",
        "",
        "The coefficient a(lambda) is fixed to a(lambda)=lambda. A neural network learns c(lambda) on the complex rectangle Re(lambda) in [-2,0], Im(lambda) in [-1,1].",
        "The analytic solution c(lambda)=lambda/(2 lambda-1) is used only for evaluation.",
        "",
        "The 3000-step entries below are checkpoints of the 10000-step run and therefore do not reproduce a separately scheduled 3000-step optimization.",
        "",
        "| method | step | c relative L2 | gap mean | gap max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, label in METHODS:
        for step, table in [(3000, metrics_3000), (10000, metrics_10000)]:
            m = table[method]
            lines.append(
                f"| {label} | {step} | {m['c_relative_l2_error']:.6e} | {m['spectral_curve_gap_mean']:.6e} | {m['spectral_curve_gap_max']:.6e} |"
            )
    lines.extend([
        "",
        "## Values reported in the collaborator note at 10000 steps",
        "",
        "| method | c relative L2 | gap mean | gap max |",
        "| --- | ---: | ---: | ---: |",
    ])
    for method, label in METHODS:
        r = REFERENCE_10000[method]
        lines.append(
            f"| {label} | {r['c_relative_l2_error']:.6e} | {r['spectral_curve_gap_mean']:.6e} | {r['spectral_curve_gap_max']:.6e} |"
        )
    lines.extend([
        "",
        "The source note mentions additional smoothness and parameter-L2 regularization terms with weight 1e-4, but does not specify the smoothness functional in enough detail to reproduce it independently. This implementation therefore compares the three reported flatness-loss constructions without those small regularizers.",
    ])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
