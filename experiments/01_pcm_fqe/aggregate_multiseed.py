from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = [
    "heldout_fqe_relative_residual",
    "q_relative_matrix_error",
    "q_offdiagonal_fraction",
    "q_diagonal_spread",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_runs(root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[-1])):
        metrics = json.loads((run_dir / "metrics.json").read_text())
        grid = np.load(run_dir / "q_grid.npz")
        runs.append(
            {
                "seed": int(metrics["seed"]),
                "metrics": metrics,
                "lam": grid["lam"],
                "q_pred": grid["q_pred"],
                "q_exact": grid["q_exact"],
                "grid_size": int(grid["grid_size"]),
            }
        )
    if not runs:
        raise RuntimeError(f"No seed_* runs found under {root}")
    return runs


def summarize(runs: list[dict]) -> dict:
    summary: dict[str, object] = {"seeds": [run["seed"] for run in runs], "n_runs": len(runs)}
    metric_summary = {}
    for key in METRIC_KEYS:
        values = np.asarray([run["metrics"][key] for run in runs], dtype=float)
        metric_summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values.tolist(),
        }
    summary["metrics"] = metric_summary

    zero_operator_norms = []
    zero_singular_values = []
    for run in runs:
        zero_entry = next(
            entry
            for entry in run["metrics"]["diagnostic_points"]
            if entry["lambda"] == [0.0, 0.0]
        )
        zero_operator_norms.append(float(zero_entry["learned_operator_norm"]))
        zero_singular_values.append([float(x) for x in zero_entry["learned_singular_values"]])

    zero_operator_norms_arr = np.asarray(zero_operator_norms)
    zero_singular_values_arr = np.asarray(zero_singular_values)
    summary["lambda_zero"] = {
        "operator_norm_mean": float(np.mean(zero_operator_norms_arr)),
        "operator_norm_std": float(np.std(zero_operator_norms_arr, ddof=1)) if len(runs) > 1 else 0.0,
        "operator_norm_values": zero_operator_norms_arr.tolist(),
        "singular_values_mean": np.mean(zero_singular_values_arr, axis=0).tolist(),
        "singular_values_std": (
            np.std(zero_singular_values_arr, axis=0, ddof=1).tolist()
            if len(runs) > 1
            else [0.0, 0.0, 0.0]
        ),
    }
    return summary


def plot_metric_by_seed(runs: list[dict], output: Path) -> None:
    seeds = [run["seed"] for run in runs]
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in METRIC_KEYS:
        values = [run["metrics"][key] for run in runs]
        ax.plot(x, values, marker="o", label=key)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("random seed")
    ax.set_ylabel("dimensionless diagnostic")
    ax.set_title("Principal chiral model: variation across independent initializations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "global_metrics_by_seed.png", dpi=180)
    plt.close(fig)


def plot_real_axis_diagonal(runs: list[dict], output: Path) -> None:
    lam = runs[0]["lam"]
    real_mask = np.isclose(lam.imag, 0.0)
    order = np.argsort(lam.real[real_mask])
    real_lam = lam.real[real_mask][order]

    learned = []
    for run in runs:
        q = run["q_pred"][real_mask][order]
        learned.append(np.trace(q, axis1=-2, axis2=-1) / 3.0)
    learned = np.asarray(learned)
    mean = np.mean(learned.real, axis=0)
    std = np.std(learned.real, axis=0, ddof=1)
    exact_q = runs[0]["q_exact"][real_mask][order]
    exact_scalar = (np.trace(exact_q, axis1=-2, axis2=-1) / 3.0).real

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(real_lam, exact_scalar, label="analytic Q scalar")
    ax.plot(real_lam, mean, label="learned mean")
    ax.fill_between(real_lam, mean - std, mean + std, alpha=0.2, label="one standard deviation")
    ax.set_xlabel("real spectral parameter")
    ax.set_ylabel("real part of mean diagonal entry")
    ax.set_title("Recovery of the scalar coefficient in Q(lambda)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "q_diagonal_real_axis_multiseed.png", dpi=180)
    plt.close(fig)


def plot_mean_error_plane(runs: list[dict], output: Path) -> None:
    lam = runs[0]["lam"]
    grid_size = runs[0]["grid_size"]
    errors = []
    for run in runs:
        diff = run["q_pred"] - run["q_exact"]
        errors.append(np.linalg.norm(diff, axis=(-2, -1)))
    mean_error = np.mean(np.asarray(errors), axis=0).reshape(grid_size, grid_size)
    re = lam.real.reshape(grid_size, grid_size)
    im = lam.imag.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    mesh = ax.pcolormesh(re, im, mean_error, shading="auto")
    fig.colorbar(mesh, ax=ax, label="mean Frobenius error")
    ax.set_xlabel("real part of spectral parameter")
    ax.set_ylabel("imaginary part of spectral parameter")
    ax.set_title("Mean pointwise Q(lambda) error across random seeds")
    fig.tight_layout()
    fig.savefig(output / "q_error_complex_plane_multiseed_mean.png", dpi=180)
    plt.close(fig)


def plot_zero_singular_values(runs: list[dict], output: Path) -> None:
    seeds = [run["seed"] for run in runs]
    x = np.arange(len(seeds))
    singular_values = []
    for run in runs:
        zero_entry = next(
            entry
            for entry in run["metrics"]["diagnostic_points"]
            if entry["lambda"] == [0.0, 0.0]
        )
        singular_values.append(zero_entry["learned_singular_values"])
    singular_values = np.asarray(singular_values)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for index in range(3):
        ax.plot(x, singular_values[:, index], marker="o", label=f"singular value {index + 1}")
    ax.set_yscale("log")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("random seed")
    ax.set_ylabel("singular value at lambda = 0")
    ax.set_title("Residual nonzero Q at the Maurer--Cartan-only point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "q_singular_values_lambda_zero_by_seed.png", dpi=180)
    plt.close(fig)


def write_markdown(summary: dict, output: Path) -> None:
    lines = [
        "# Experiment 01 multi-seed summary",
        "",
        f"Independent runs: {summary['n_runs']}",
        f"Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "",
        "| diagnostic | mean | standard deviation | minimum | maximum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, values in summary["metrics"].items():
        lines.append(
            f"| {key} | {values['mean']:.6e} | {values['std']:.6e} | {values['min']:.6e} | {values['max']:.6e} |"
        )
    zero = summary["lambda_zero"]
    lines.extend(
        [
            "",
            "At spectral parameter lambda = 0 the exact map Q vanishes. The learned operator norm therefore measures approximation error rather than a nonzero rank:",
            "",
            f"- mean learned operator norm: {zero['operator_norm_mean']:.6e}",
            f"- standard deviation: {zero['operator_norm_std']:.6e}",
            "",
            "Generated figures:",
            "",
            "- global_metrics_by_seed.png",
            "- q_diagonal_real_axis_multiseed.png",
            "- q_error_complex_plane_multiseed_mean.png",
            "- q_singular_values_lambda_zero_by_seed.png",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_metric_by_seed(runs, args.output)
    plot_real_axis_diagonal(runs, args.output)
    plot_mean_error_plane(runs, args.output)
    plot_zero_singular_values(runs, args.output)
    write_markdown(summary, args.output)


if __name__ == "__main__":
    main()
