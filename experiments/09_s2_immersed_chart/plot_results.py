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


def heatmap(ax, values, s_values, t_values, title, logarithmic=False):
    shown = np.log10(np.maximum(values, 1.0e-16)) if logarithmic else values
    image = ax.imshow(
        shown.T,
        origin="lower",
        aspect="auto",
        extent=[s_values[0], s_values[-1], t_values[0], t_values[-1]],
    )
    ax.set_xlabel(r"anchor arclength coordinate $s$")
    ax.set_ylabel(r"transverse coordinate $t$")
    ax.set_title(title)
    return image


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.results / "history.json").read_text())
    with np.load(args.results / "evaluation.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}

    steps = np.asarray([row["step"] for row in history])
    train_flat = np.asarray([row["training"]["flatness_loss"] for row in history])
    val_flat = np.asarray([row["validation"]["flatness_loss"] for row in history])
    train_total = np.asarray([row["training"]["total_loss"] for row in history])
    val_total = np.asarray([row["validation"]["total_loss"] for row in history])
    widths = np.asarray([row["training_half_width"] for row in history])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(steps, np.maximum(train_flat, 1.0e-16), label="training flatness")
    ax.plot(steps, np.maximum(val_flat, 1.0e-16), label="validation flatness")
    ax.plot(steps, np.maximum(train_total / 100.0, 1.0e-16), label="training total / 100")
    ax.plot(steps, np.maximum(val_total / 100.0, 1.0e-16), label="validation total / 100")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "training_and_validation_losses.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.step(steps, widths, where="post")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"sampled transverse half-width $|t|_{\max}$")
    fig.tight_layout()
    fig.savefig(args.output / "transverse_width_curriculum.png", dpi=180)
    plt.close(fig)

    s_values = data["s_values"]
    t_values = data["t_values"]
    for key, filename, title, label, logarithmic in [
        ("flatness", "flatness_heatmap.png", "held-out flatness", r"$\log_{10}$ component-normalized flatness", True),
        ("jacobian_sigma_ratio", "chart_jacobian_sigma_ratio.png", "chart Jacobian singular-value ratio", r"$\sigma_{\min}/\sigma_{\max}$", False),
    ]:
        fig, ax = plt.subplots(figsize=(7.3, 4.5))
        image = heatmap(ax, data[key], s_values, t_values, title, logarithmic)
        fig.colorbar(image, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(args.output / filename, dpi=180)
        plt.close(fig)

    singular_values = data["jacobian_singular_values"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    image0 = heatmap(axes[0], singular_values[..., 0], s_values, t_values, r"$\sigma_{\max}$")
    image1 = heatmap(axes[1], singular_values[..., 1], s_values, t_values, r"$\sigma_{\min}$")
    fig.colorbar(image0, ax=axes[0])
    fig.colorbar(image1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(args.output / "chart_jacobian_singular_values.png", dpi=180)
    plt.close(fig)

    b = data["coefficients_real"][..., 1] + 1j * data["coefficients_imag"][..., 1]
    colors = np.broadcast_to(t_values[None, :], b.shape)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    scatter = ax.scatter(b.real.ravel(), b.imag.ravel(), c=colors.ravel(), s=7)
    ax.set_xlabel(r"$\operatorname{Re} b$")
    ax.set_ylabel(r"$\operatorname{Im} b$")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(scatter, ax=ax, label=r"transverse coordinate $t$")
    fig.tight_layout()
    fig.savefig(args.output / "b_complex_plane.png", dpi=180)
    plt.close(fig)

    invariant_error = np.maximum.reduce(
        [data["abs_a_minus_one"], data["abs_c_minus_one"], data["abs_bd_minus_one"]]
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    image = heatmap(
        ax,
        invariant_error,
        s_values,
        t_values,
        "analytic-family validation",
        logarithmic=True,
    )
    fig.colorbar(image, ax=ax, label=r"$\log_{10}\max(|a-1|,|c-1|,|bd-1|)$")
    fig.tight_layout()
    fig.savefig(args.output / "analytic_family_invariant_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    axes[0].plot(data["anchor_frame_s"], data["tangent_s_cosine"])
    axes[1].plot(data["anchor_frame_s"], data["tangent_t_cosine"])
    axes[0].set_ylabel(r"$\widehat{\partial_sN}\cdot v_s$")
    axes[1].set_ylabel(r"$\widehat{\partial_tN}\cdot v_t$")
    axes[1].set_xlabel(r"anchor arclength coordinate $s$")
    for ax in axes:
        ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(args.output / "anchor_tangent_alignment.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
