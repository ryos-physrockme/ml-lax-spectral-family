from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from ml_lax_spectral_family.models import ComplexLinearMapNet


METHODS = ("plain", "analytic_weight", "normalized_adaptive")
LAMBDA_SAMPLING_MODES = ("fixed", "resampled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--lambda-sampling",
        choices=LAMBDA_SAMPLING_MODES,
        default="fixed",
        help=(
            "fixed: sample the spectral-parameter batch once per run; "
            "resampled: draw a new spectral-parameter batch at every optimization step"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. By default use output_root/method from the YAML file.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def learning_rate(step: int, total_steps: int, peak: float, minimum: float, warmup: int) -> float:
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def sample_lambda(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    domain = cfg["domain"]
    real = rng.uniform(domain["re_min"], domain["re_max"], size=n)
    imag = rng.uniform(domain["im_min"], domain["im_max"], size=n)
    return real + 1j * imag


def analytic_c(lam: np.ndarray) -> np.ndarray:
    return lam / (2.0 * lam - 1.0)


def evaluate(model: torch.nn.Module, cfg: dict, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    ecfg = cfg["evaluation"]
    domain = cfg["domain"]
    n = int(ecfg["grid_size"])
    re = np.linspace(domain["re_min"], domain["re_max"], n)
    im = np.linspace(domain["im_min"], domain["im_max"], n)
    rr, ii = np.meshgrid(re, im, indexing="xy")
    lam = (rr + 1j * ii).reshape(-1)
    xy = torch.as_tensor(np.column_stack([lam.real, lam.imag]), dtype=torch.float64, device=device)
    with torch.no_grad():
        c_pred = model(xy)[..., 0, 0].cpu().numpy()
    c_exact = analytic_c(lam)
    gap = np.abs(lam + c_pred - 2.0 * lam * c_pred)
    rel_l2 = float(np.linalg.norm(c_pred - c_exact) / np.linalg.norm(c_exact))

    n_real = int(ecfg["real_axis_points"])
    lam_real = np.linspace(domain["re_min"], domain["re_max"], n_real).astype(np.complex128)
    xy_real = torch.as_tensor(
        np.column_stack([lam_real.real, lam_real.imag]), dtype=torch.float64, device=device
    )
    with torch.no_grad():
        c_real = model(xy_real)[..., 0, 0].cpu().numpy()

    metrics = {
        "c_relative_l2_error": rel_l2,
        "spectral_curve_gap_mean": float(np.mean(gap)),
        "spectral_curve_gap_max": float(np.max(gap)),
    }
    arrays = {
        "lam": lam,
        "c_pred": c_pred,
        "c_exact": c_exact,
        "gap": gap,
        "grid_size": np.asarray(n),
        "lam_real": lam_real,
        "c_real": c_real,
        "c_real_exact": analytic_c(lam_real),
    }
    return metrics, arrays


def lambda_tensors(
    cfg: dict,
    n_lambda: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    lam_np = sample_lambda(cfg, n_lambda, rng)
    lam_xy = torch.as_tensor(
        np.column_stack([lam_np.real, lam_np.imag]), dtype=torch.float64, device=device
    )
    lam = torch.as_tensor(lam_np, dtype=torch.complex128, device=device)
    return lam_np, lam_xy, lam


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    method = args.method
    lambda_sampling = args.lambda_sampling
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(args.device)

    tcfg = cfg["training"]
    model = ComplexLinearMapNet(
        output_rows=1,
        output_cols=1,
        hidden_dim=int(tcfg["hidden_dim"]),
        hidden_layers=int(tcfg["hidden_layers"]),
    ).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["peak_learning_rate"]))

    total_steps = int(tcfg["steps"])
    n_lambda = int(tcfg["n_lambda"])
    n_field = int(tcfg["n_field_samples"])
    checkpoint_steps = {int(value) for value in tcfg["checkpoint_steps"]}
    history_every = int(tcfg["history_every"])
    history: list[dict] = []

    # The collaborator note defines an exponential moving average indexed by
    # lambda_n but does not explicitly state whether the spectral-parameter
    # batch is held fixed or redrawn at every optimization step.  Both choices
    # are supported here so that the ambiguity can be measured rather than
    # silently resolved.
    fixed_lam_np: np.ndarray | None = None
    fixed_lam_xy: torch.Tensor | None = None
    fixed_lam: torch.Tensor | None = None
    if lambda_sampling == "fixed":
        fixed_lam_np, fixed_lam_xy, fixed_lam = lambda_tensors(cfg, n_lambda, rng, device)

    acfg = cfg["adaptive_weighting"]
    ema_alpha = float(acfg["exponential_moving_average_alpha"])
    clip_ratio = float(acfg["clipping_ratio"])
    adaptive_warmup = int(acfg["warmup_steps"])
    denominator_epsilon = float(acfg["denominator_epsilon"])
    ema_loss: torch.Tensor | None = None

    if args.output_dir is None:
        output_dir = Path(cfg["output_root"]) / method
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixed_lam_np is not None:
        np.save(output_dir / "training_lambda.npy", fixed_lam_np)

    run_metadata = {
        "method": method,
        "lambda_sampling": lambda_sampling,
        "seed": seed,
        "n_lambda": n_lambda,
        "n_field_samples_per_lambda": n_field,
        "adaptive_ema_interpretation": (
            "per fixed spectral-parameter sample"
            if lambda_sampling == "fixed"
            else "per batch index; the spectral-parameter value at that index changes between steps"
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    for step in range(1, total_steps + 1):
        lr = learning_rate(
            step,
            total_steps,
            float(tcfg["peak_learning_rate"]),
            float(tcfg["minimum_learning_rate"]),
            int(tcfg["warmup_steps"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if lambda_sampling == "fixed":
            assert fixed_lam_np is not None and fixed_lam_xy is not None and fixed_lam is not None
            lam_np, lam_xy, lam = fixed_lam_np, fixed_lam_xy, fixed_lam
        else:
            lam_np, lam_xy, lam = lambda_tensors(cfg, n_lambda, rng, device)

        a = lam
        c = model(lam_xy)[..., 0, 0]

        j_plus = rng.normal(size=(n_lambda, n_field, 3))
        j_minus = rng.normal(size=(n_lambda, n_field, 3))
        bracket_np = np.cross(j_plus, j_minus)
        bracket = torch.as_tensor(bracket_np, dtype=torch.float64, device=device)

        curve = a + c - 2.0 * a * c
        curvature = -0.5 * curve[:, None, None] * bracket
        curvature_sq = torch.sum(torch.abs(curvature) ** 2, dim=-1)

        if method in ("plain", "analytic_weight"):
            per_lambda = torch.mean(curvature_sq, dim=1)
            if method == "plain":
                weights = torch.ones_like(per_lambda)
            else:
                weights = torch.abs(lam - 0.5).pow(-2).real
                weights = weights / torch.mean(weights)
            loss = torch.mean(weights * per_lambda)
        else:
            term_1 = -0.5 * c[:, None, None] * bracket
            term_2 = 0.5 * a[:, None, None] * bracket
            term_3 = (a * c)[:, None, None] * bracket
            denominator = (
                torch.sum(torch.abs(term_1) ** 2, dim=-1)
                + torch.sum(torch.abs(term_2) ** 2, dim=-1)
                + torch.sum(torch.abs(term_3) ** 2, dim=-1)
                + denominator_epsilon
            )
            normalized_sample_loss = curvature_sq / denominator
            per_lambda = torch.mean(normalized_sample_loss, dim=1)
            detached = per_lambda.detach()
            if ema_loss is None:
                ema_loss = detached
            else:
                ema_loss = ema_alpha * ema_loss + (1.0 - ema_alpha) * detached
            if step <= adaptive_warmup:
                weights = torch.ones_like(per_lambda)
            else:
                ratio = ema_loss / torch.mean(ema_loss)
                weights = torch.clamp(ratio, 1.0 / clip_ratio, clip_ratio)
                weights = weights / torch.mean(weights)
            loss = torch.mean(weights.detach() * per_lambda)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % history_every == 0 or step == total_steps:
            history.append(
                {
                    "step": step,
                    "training_loss": float(loss.item()),
                    "learning_rate": lr,
                }
            )
        if step == 1 or step % int(tcfg["log_every"]) == 0:
            print(
                f"method={method} lambda_sampling={lambda_sampling} "
                f"step={step:6d} loss={loss.item():.6e} lr={lr:.3e}"
            )

        if step in checkpoint_steps:
            metrics, arrays = evaluate(model, cfg, device)
            metrics.update(
                {
                    "method": method,
                    "lambda_sampling": lambda_sampling,
                    "step": step,
                    "seed": seed,
                }
            )
            (output_dir / f"metrics_step_{step}.json").write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(output_dir / f"grid_step_{step}.npz", **arrays)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), output_dir / "c_net.pt")


if __name__ == "__main__":
    main()
