from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("plain", "Bare flatness loss"),
    ("analytic_weight", "Analytic per-spectral-parameter weight"),
    ("normalized_adaptive", "Component normalization + adaptive weight"),
]
SAMPLING_MODES = [
    ("fixed", "Fixed spectral-parameter batch"),
    ("resampled", "Resampled spectral-parameter batch"),
]
REFERENCE = {
    "plain": {
        "c_relative_l2_error": 1.7e-2,
        "spectral_curve_gap_mean": 5.9e-3,
        "spectral_curve_gap_max": 7.0e-2,
    },
    "analytic_weight": {
        "c_relative_l2_error": 7.6e-3,
        "spectral_curve_gap_mean": 5.9e-3,
        "spectral_curve_gap_max": 3.0e-2,
    },
    "normalized_adaptive": {
        "c_relative_l2_error": 1.1e-2,
        "spectral_curve_gap_mean": 9.7e-3,
        "spectral_curve_gap_max": 5.5e-2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(root: Path, sampling: str, method: str) -> dict:
    path = root / sampling / method / "metrics_step_10000.json"
    return json.loads(path.read_text())


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def grouped_metric_plot(
    root: Path,
    output: Path,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    x = np.arange(len(METHODS))
    width = 0.25
    fixed = [load_metrics(root, "fixed", method)[metric] for method, _ in METHODS]
    resampled = [load_metrics(root, "resampled", method)[metric] for method, _ in METHODS]
    reference = [REFERENCE[method][metric] for method, _ in METHODS]

    plt.figure(figsize=(8.3, 4.8))
    plt.bar(x - width, fixed, width, label="Fixed spectral-parameter batch")
    plt.bar(x, resampled, width, label="Resampled spectral-parameter batch")
    plt.bar(x + width, reference, width, label="Collaborator note")
    plt.xticks(x, ["Bare", "Analytic weight", "Normalized + adaptive"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    save(output / filename)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grouped_metric_plot(
        args.results,
        args.output,
        "c_relative_l2_error",
        r"Relative $L^2$ error of $c(\lambda)$",
        "Dependence on spectral-parameter sampling cadence",
        "relative_l2_sampling_ablation.png",
    )
    grouped_metric_plot(
        args.results,
        args.output,
        "spectral_curve_gap_max",
        r"Maximum $|a+c-2ac|$",
        "Worst spectral-curve error and sampling cadence",
        "gap_max_sampling_ablation.png",
    )

    plt.figure(figsize=(7.6, 4.6))
    for sampling, sampling_label in SAMPLING_MODES:
        path = args.results / sampling / "normalized_adaptive" / "history.json"
        history = json.loads(path.read_text())
        steps = [row["step"] for row in history]
        losses = [row["training_loss"] for row in history]
        plt.semilogy(steps, losses, label=sampling_label)
    plt.xlabel("Training step")
    plt.ylabel("Normalized adaptive training objective")
    plt.title("Adaptive loss under two spectral-parameter sampling conventions")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save(args.output / "adaptive_training_objective_sampling_ablation.png")

    lines = [
        "# Experiment 04: spectral-parameter sampling-cadence ablation",
        "",
        "This experiment repeats the three principal-chiral-model hard-parametrization loss constructions under two choices for the spectral-parameter batch.",
        "",
        "- `fixed`: the 64 spectral-parameter values are sampled once and kept for the entire run; the exponential moving average in the adaptive method follows a fixed spectral-parameter value.",
        "- `resampled`: 64 new spectral-parameter values are drawn at every optimization step; the exponential moving average follows a batch index rather than a persistent point of the spectral-parameter plane.",
        "",
        "The collaborator note specifies uniform spectral-parameter sampling and an exponential moving average indexed by lambda_n, but it does not explicitly specify the resampling cadence. The two conventions are therefore treated as an implementation ablation rather than as two claims about the collaborator code.",
        "",
        "| sampling | method | c relative L2 | gap mean | gap max |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for sampling, _ in SAMPLING_MODES:
        for method, method_label in METHODS:
            m = load_metrics(args.results, sampling, method)
            lines.append(
                f"| {sampling} | {method_label} | {m['c_relative_l2_error']:.6e} | {m['spectral_curve_gap_mean']:.6e} | {m['spectral_curve_gap_max']:.6e} |"
            )
    lines.extend(
        [
            "",
            "## Collaborator-note values at 10000 steps",
            "",
            "| method | c relative L2 | gap mean | gap max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method, method_label in METHODS:
        r = REFERENCE[method]
        lines.append(
            f"| {method_label} | {r['c_relative_l2_error']:.6e} | {r['spectral_curve_gap_mean']:.6e} | {r['spectral_curve_gap_max']:.6e} |"
        )
    lines.extend(
        [
            "",
            "The purpose of this ablation is to determine which conclusions are stable under an implementation detail that is not fixed by the available note. In particular, the relative ordering of the analytic weighting and the adaptive weighting should not be treated as reproduced unless it is stable or the original sampling cadence is known.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
