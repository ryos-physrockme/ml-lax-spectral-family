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


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def offdiagonal_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    diag_only = np.zeros_like(q)
    idx = np.arange(q.shape[-1])
    diag_only[..., idx, idx] = q[..., idx, idx]
    return np.linalg.norm(q - diag_only, axis=(-2, -1))


def main() -> None:
    args = parse_args()
    data = np.load(args.results / "q_grid.npz")
    history = json.loads((args.results / "history.json").read_text())
    metrics = json.loads((args.results / "metrics.json").read_text())

    lam = data["lam"]
    q_pred = data["q_pred"]
    q_exact = data["q_exact"]
    grid_size = int(data["grid_size"])

    re_values = np.unique(lam.real)
    im_values = np.unique(lam.imag)
    if len(re_values) != grid_size or len(im_values) != grid_size:
        raise RuntimeError("The saved spectral-parameter grid is not a square tensor-product grid.")

    steps = np.asarray([item["step"] for item in history])
    losses = np.asarray([item["normalized_fqe_loss"] for item in history])
    plt.figure(figsize=(6.2, 4.2))
    plt.semilogy(steps, losses)
    plt.xlabel("Training step")
    plt.ylabel(r"Normalized $F_{+-}-Q E$ loss")
    plt.title("Principal chiral model: training history")
    plt.grid(True, alpha=0.25)
    save_figure(args.output / "training_loss.png")

    zero_im_index = int(np.argmin(np.abs(im_values)))
    row = slice(zero_im_index * grid_size, (zero_im_index + 1) * grid_size)
    lam_real = lam[row].real
    q_pred_real = q_pred[row]
    q_exact_real = q_exact[row]
    learned_diag_mean = np.mean(np.diagonal(q_pred_real, axis1=-2, axis2=-1), axis=-1)
    exact_diag = np.diagonal(q_exact_real, axis1=-2, axis2=-1)[..., 0]

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(lam_real, exact_diag.real, linestyle="--", label="Exact real part")
    plt.plot(lam_real, learned_diag_mean.real, label="Learned diagonal mean: real part")
    plt.plot(lam_real, learned_diag_mean.imag, label="Learned diagonal mean: imaginary part")
    plt.xlabel(r"Real spectral parameter $\lambda$ with $\operatorname{Im}\lambda=0$")
    plt.ylabel(r"Mean diagonal entry of $Q(\lambda)$")
    plt.title("Learned equation-of-motion map on the real spectral-parameter axis")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(args.output / "q_diagonal_real_axis.png")

    offdiag_real = offdiagonal_norm(q_pred_real)
    plt.figure(figsize=(6.2, 4.2))
    plt.semilogy(lam_real, np.maximum(offdiag_real, 1.0e-16))
    plt.xlabel(r"Real spectral parameter $\lambda$ with $\operatorname{Im}\lambda=0$")
    plt.ylabel(r"Frobenius norm of off-diagonal entries of $Q(\lambda)$")
    plt.title("Emergent suppression of off-diagonal matrix entries")
    plt.grid(True, alpha=0.25)
    save_figure(args.output / "q_offdiagonal_real_axis.png")

    pointwise_error = np.linalg.norm(q_pred - q_exact, axis=(-2, -1)).reshape(grid_size, grid_size)
    plt.figure(figsize=(6.1, 5.0))
    image = plt.imshow(
        np.log10(np.maximum(pointwise_error, 1.0e-16)),
        origin="lower",
        extent=[re_values.min(), re_values.max(), im_values.min(), im_values.max()],
        aspect="equal",
    )
    plt.colorbar(image, label=r"$\log_{10}\|Q_{\rm learned}-Q_{\rm exact}\|_F$")
    plt.xlabel(r"$\operatorname{Re}\lambda$")
    plt.ylabel(r"$\operatorname{Im}\lambda$")
    plt.title("Pointwise error over the complex spectral-parameter domain")
    save_figure(args.output / "q_error_complex_plane.png")

    singular_values = np.linalg.svd(q_pred_real, compute_uv=False)
    exact_scalar_magnitude = np.abs(exact_diag)
    plt.figure(figsize=(6.2, 4.2))
    for index in range(singular_values.shape[1]):
        plt.plot(lam_real, singular_values[:, index], label=fr"Learned $\sigma_{index + 1}$")
    plt.plot(lam_real, exact_scalar_magnitude, linestyle="--", label=r"Exact $|q(\lambda)|$")
    plt.xlabel(r"Real spectral parameter $\lambda$ with $\operatorname{Im}\lambda=0$")
    plt.ylabel(r"Singular value of $Q(\lambda)$")
    plt.title("Rank diagnostic from singular values")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure(args.output / "q_singular_values_real_axis.png")

    summary_lines = [
        "# Experiment 01 numerical summary",
        "",
        "The neural network was trained to learn a general complex 3 x 3 matrix Q(lambda) from off-shell principal-chiral-model samples satisfying the Maurer--Cartan identity.",
        "The analytic form of Q(lambda) was used only after training for validation.",
        "",
        f"- Held-out relative residual ||F-QE||/||F||: {metrics['heldout_fqe_relative_residual']:.6e}",
        f"- Relative matrix error for Q: {metrics['q_relative_matrix_error']:.6e}",
        f"- Fraction of matrix norm in off-diagonal entries: {metrics['q_offdiagonal_fraction']:.6e}",
        f"- Relative spread among diagonal entries: {metrics['q_diagonal_spread']:.6e}",
        f"- Rank of vertically stacked exact Q matrices at diagnostic points: {metrics['exact_stacked_rank_diagnostic_points']}",
        f"- Rank of vertically stacked learned Q matrices at diagnostic points: {metrics['learned_stacked_rank_diagnostic_points']}",
        "",
        "Generated figures:",
        "",
        "- training_loss.png",
        "- q_diagonal_real_axis.png",
        "- q_offdiagonal_real_axis.png",
        "- q_error_complex_plane.png",
        "- q_singular_values_real_axis.png",
    ]
    (args.output / "summary.md").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
