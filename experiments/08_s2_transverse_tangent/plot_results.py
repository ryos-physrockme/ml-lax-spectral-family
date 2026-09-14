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


def load_variant(results: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(results / f"evaluation_{name}.npz") as data:
        return {key: data[key] for key in data.files}


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


def plot_training(results: Path, output: Path, names: list[str]) -> None:
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 6.4), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        history = json.loads((results / f"history_{name}.json").read_text())
        steps = np.asarray([row["step"] for row in history])
        for key, label in [
            ("flatness_loss", "flatness"),
            ("anchor_loss", "anchor"),
            ("derivative_norm_loss", "derivative norm"),
            ("barrier_loss", "log rank barrier"),
            ("frame_loss", "tangent frame"),
        ]:
            values = np.asarray([row[key] for row in history])
            ax.plot(steps, np.maximum(values, 1.0e-16), label=label)
        ax.set_yscale("log")
        ax.set_ylabel(name.replace("_", " "))
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)


def plot_grid_metric(output: Path, variants, key, filename, colorbar_label, logarithmic=False):
    fig, axes = plt.subplots(1, len(variants), figsize=(10.4, 4.1), squeeze=False)
    for ax, (name, data) in zip(axes[0], variants.items()):
        image = heatmap(
            ax,
            data[key],
            data["s_values"],
            data["t_values"],
            name.replace("_", " "),
            logarithmic=logarithmic,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def plot_tangent_alignment(output: Path, variants) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    for name, data in variants.items():
        s = data["anchor_frame_s"]
        axes[0].plot(s, data["tangent_s_cosine"], label=name.replace("_", " "))
        axes[1].plot(s, data["tangent_t_cosine"], label=name.replace("_", " "))
    axes[0].set_ylabel(r"$\widehat{\partial_sN}\cdot v_s$")
    axes[1].set_ylabel(r"$\widehat{\partial_tN}\cdot v_t$")
    axes[1].set_xlabel(r"anchor arclength coordinate $s$")
    for ax in axes:
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "anchor_tangent_alignment.png", dpi=180)
    plt.close(fig)


def plot_b_plane(output: Path, variants) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(10.4, 4.4), squeeze=False)
    for ax, (name, data) in zip(axes[0], variants.items()):
        b = data["coefficients_real"][..., 1] + 1j * data["coefficients_imag"][..., 1]
        t_values = data["t_values"]
        colors = np.broadcast_to(t_values[None, :], b.shape)
        scatter = ax.scatter(b.real.ravel(), b.imag.ravel(), c=colors.ravel(), s=7)
        ax.set_xlabel(r"$\operatorname{Re} b$")
        ax.set_ylabel(r"$\operatorname{Im} b$")
        ax.set_title(name.replace("_", " "))
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=ax, label=r"transverse coordinate $t$")
    fig.tight_layout()
    fig.savefig(output / "b_complex_plane.png", dpi=180)
    plt.close(fig)


def plot_invariant_error(output: Path, variants) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(10.4, 4.1), squeeze=False)
    for ax, (name, data) in zip(axes[0], variants.items()):
        combined = np.maximum.reduce(
            [data["abs_a_minus_one"], data["abs_c_minus_one"], data["abs_bd_minus_one"]]
        )
        image = heatmap(
            ax,
            combined,
            data["s_values"],
            data["t_values"],
            name.replace("_", " "),
            logarithmic=True,
        )
        fig.colorbar(image, ax=ax, label=r"$\log_{10}\max(|a-1|,|c-1|,|bd-1|)$")
    fig.tight_layout()
    fig.savefig(output / "analytic_family_invariant_error.png", dpi=180)
    plt.close(fig)


def plot_anchor_line_rank(output: Path, variants) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    for name, data in variants.items():
        ax.plot(
            data["anchor_frame_s"],
            data["anchor_line_sigma_ratio"],
            label=name.replace("_", " "),
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"anchor arclength coordinate $s$")
    ax.set_ylabel(r"$\sigma_{\min}(J_N)/\sigma_{\max}(J_N)$ on $t=0$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "anchor_line_jacobian_rank.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "summary.json").read_text())
    names = [
        name
        for name in summary
        if name not in {"tangent_frame_construction", "chart_coordinate"}
    ]
    variants = {name: load_variant(args.results, name) for name in names}

    plot_training(args.results, args.output, names)
    plot_grid_metric(
        args.output,
        variants,
        "flatness",
        "flatness_heatmap.png",
        r"$\log_{10}$ component-normalized flatness",
        logarithmic=True,
    )
    plot_grid_metric(
        args.output,
        variants,
        "jacobian_sigma_ratio",
        "chart_jacobian_sigma_ratio.png",
        r"$\sigma_{\min}(J_N)/\sigma_{\max}(J_N)$",
    )
    plot_grid_metric(
        args.output,
        variants,
        "sine_squared",
        "chart_tangent_sine_squared.png",
        r"$\sin^2\angle(\partial_sN,\partial_tN)$",
    )
    plot_tangent_alignment(args.output, variants)
    plot_anchor_line_rank(args.output, variants)
    plot_b_plane(args.output, variants)
    plot_invariant_error(args.output, variants)


if __name__ == "__main__":
    main()
