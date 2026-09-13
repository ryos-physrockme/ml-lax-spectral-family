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


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((args.results / "metrics.json").read_text())
    history = json.loads((args.results / "history.json").read_text())
    grid = np.load(args.results / "q_grid.npz")
    lam = grid["lam"]
    q_pred = grid["q_pred"]
    q_exact = grid["q_exact"]
    grid_size = int(grid["grid_size"])

    steps = np.asarray([entry["step"] for entry in history])
    loss = np.asarray([entry["normalized_fqe_loss"] for entry in history])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(steps, loss)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("normalized squared residual")
    ax.set_title("S^2 sigma model: training history for F = Q E")
    fig.tight_layout()
    fig.savefig(args.output / "training_loss.png", dpi=180)
    plt.close(fig)

    real_mask = np.isclose(lam.imag, 0.0)
    order = np.argsort(lam.real[real_mask])
    real_lam = lam.real[real_mask][order]
    q_real = q_pred[real_mask][order]
    q_exact_real = q_exact[real_mask][order]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(real_lam, q_exact_real[:, 0, 0].real, label="analytic Q[0,0]")
    ax.plot(real_lam, q_real[:, 0, 0].real, label="learned Q[0,0]")
    ax.plot(real_lam, q_exact_real[:, 1, 1].real, label="analytic Q[1,1]")
    ax.plot(real_lam, q_real[:, 1, 1].real, label="learned Q[1,1]")
    ax.set_xlabel("real spectral parameter")
    ax.set_ylabel("real matrix entry")
    ax.set_title("Active entries of the learned 3 x 2 map")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "q_active_entries_real_axis.png", dpi=180)
    plt.close(fig)

    allowed = np.zeros_like(q_real)
    allowed[:, 0, 0] = q_real[:, 0, 0]
    allowed[:, 1, 1] = q_real[:, 1, 1]
    forbidden = q_real - allowed
    forbidden_norm = np.linalg.norm(forbidden, axis=(-2, -1))
    total_norm = np.linalg.norm(q_real, axis=(-2, -1))
    fraction = forbidden_norm / (total_norm + 1.0e-30)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(real_lam, fraction)
    ax.set_yscale("log")
    ax.set_xlabel("real spectral parameter")
    ax.set_ylabel("forbidden-entry norm fraction")
    ax.set_title("Entries not present in the analytic linear map")
    fig.tight_layout()
    fig.savefig(args.output / "q_forbidden_entries_real_axis.png", dpi=180)
    plt.close(fig)

    diff = q_pred - q_exact
    point_error = np.linalg.norm(diff, axis=(-2, -1)).reshape(grid_size, grid_size)
    re = lam.real.reshape(grid_size, grid_size)
    im = lam.imag.reshape(grid_size, grid_size)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    mesh = ax.pcolormesh(re, im, point_error, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Frobenius error")
    ax.set_xlabel("real part of spectral parameter")
    ax.set_ylabel("imaginary part of spectral parameter")
    ax.set_title("Pointwise error of the learned Q(lambda)")
    fig.tight_layout()
    fig.savefig(args.output / "q_error_complex_plane.png", dpi=180)
    plt.close(fig)

    learned_sv = np.linalg.svd(q_real, compute_uv=False)
    exact_sv = np.linalg.svd(q_exact_real, compute_uv=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(real_lam, learned_sv[:, 0], label="learned singular value 1")
    ax.plot(real_lam, learned_sv[:, 1], label="learned singular value 2")
    ax.plot(real_lam, exact_sv[:, 0], linestyle="--", label="analytic singular value")
    ax.set_xlabel("real spectral parameter")
    ax.set_ylabel("singular value")
    ax.set_title("Two equation-of-motion directions encoded by Q(lambda)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "q_singular_values_real_axis.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Experiment 02 numerical summary",
        "",
        "The neural network maps the complex spectral parameter to a general complex 3 x 2 matrix.",
        "The exact sparse matrix structure is not imposed during training.",
        "",
        f"- Held-out relative residual ||F-QE||/||F||: {metrics['heldout_fqe_relative_residual']:.6e}",
        f"- Relative matrix error for Q: {metrics['q_relative_matrix_error']:.6e}",
        f"- Fraction of matrix norm in analytically forbidden entries: {metrics['q_forbidden_entry_fraction']:.6e}",
        f"- Relative violation of Q[0,0] + Q[1,1] = 0: {metrics['q_active_entry_relation_error']:.6e}",
        "",
        "Generated figures:",
        "",
        "- training_loss.png",
        "- q_active_entries_real_axis.png",
        "- q_forbidden_entries_real_axis.png",
        "- q_error_complex_plane.png",
        "- q_singular_values_real_axis.png",
        "",
    ]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
