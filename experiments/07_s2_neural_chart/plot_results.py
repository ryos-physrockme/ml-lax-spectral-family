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


def plot_training_histories(results: Path, output: Path, names: list[str]) -> None:
    fig, axes = plt.subplots(len(names), 1, figsize=(7.0, 6.2), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        history = json.loads((results / f"history_{name}.json").read_text())
        steps = np.asarray([row["step"] for row in history])
        for key, label in [
            ("flatness_loss", "flatness"),
            ("anchor_loss", "anchor"),
            ("derivative_norm_loss", "derivative norm"),
            ("rank_loss", "rank"),
        ]:
            values = np.asarray([row[key] for row in history])
            positive = np.maximum(values, 1.0e-16)
            ax.plot(steps, positive, label=label)
        ax.set_yscale("log")
        ax.set_ylabel(name.replace("_", " "))
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)


def plot_b_plane(output: Path, variants: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(10.0, 4.4), squeeze=False)
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


def heatmap(
    ax,
    data: np.ndarray,
    s_values: np.ndarray,
    t_values: np.ndarray,
    title: str,
    logarithmic: bool = False,
):
    shown = np.log10(np.maximum(data, 1.0e-16)) if logarithmic else data
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


def plot_heatmap_comparison(
    output: Path,
    variants: dict[str, dict[str, np.ndarray]],
    key: str,
    filename: str,
    label: str,
    logarithmic: bool,
) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(10.0, 4.0), squeeze=False)
    for ax, (name, data) in zip(axes[0], variants.items()):
        image = heatmap(
            ax,
            data[key],
            data["s_values"],
            data["t_values"],
            name.replace("_", " "),
            logarithmic=logarithmic,
        )
        fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def plot_invariant_error(output: Path, variants: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, len(variants), figsize=(10.0, 4.0), squeeze=False)
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


def plot_anchor_fit(output: Path, variants: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    first = next(iter(variants.values()))
    anchor_s = first["anchor_s"]
    anchor_b = first["anchor_coefficients_real"][:, 1] + 1j * first["anchor_coefficients_imag"][:, 1]
    ax.plot(anchor_s, np.abs(anchor_b), "o", ms=3, label="continuation anchors")
    for name, data in variants.items():
        prediction = data["anchor_prediction"]
        b_prediction = prediction[:, 1] + 1j * prediction[:, 5]
        ax.plot(anchor_s, np.abs(b_prediction), label=name.replace("_", " "))
    ax.set_xlabel(r"anchor arclength coordinate $s$")
    ax.set_ylabel(r"$|b(s,0)|$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "anchor_fit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "summary.json").read_text())
    names = [name for name in summary if name != "chart_coordinate"]
    variants = {name: load_variant(args.results, name) for name in names}

    plot_training_histories(args.results, args.output, names)
    plot_b_plane(args.output, variants)
    plot_heatmap_comparison(
        args.output,
        variants,
        key="flatness",
        filename="flatness_heatmap.png",
        label=r"$\log_{10}$ component-normalized flatness",
        logarithmic=True,
    )
    plot_heatmap_comparison(
        args.output,
        variants,
        key="jacobian_sigma_ratio",
        filename="chart_jacobian_sigma_ratio.png",
        label=r"$\sigma_{\min}(J_N)/\sigma_{\max}(J_N)$",
        logarithmic=False,
    )
    plot_heatmap_comparison(
        args.output,
        variants,
        key="sine_squared",
        filename="chart_tangent_sine_squared.png",
        label=r"$\sin^2\angle(\partial_s N,\partial_t N)$",
        logarithmic=False,
    )
    plot_invariant_error(args.output, variants)
    plot_anchor_fit(args.output, variants)


if __name__ == "__main__":
    main()
