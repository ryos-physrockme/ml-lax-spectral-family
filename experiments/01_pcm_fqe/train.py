from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ml_lax_spectral_family.diagnostics import (
    diagonal_spread,
    offdiagonal_fraction,
    relative_matrix_error,
    relative_residual,
    singular_values,
)
from ml_lax_spectral_family.models import ComplexMatrixNet
from ml_lax_spectral_family.pcm import (
    apply_q,
    exact_q_matrix,
    sample_offshell_batch,
    stacked_q_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sample_lambda(cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    d = cfg["domain"]
    re = rng.uniform(d["re_min"], d["re_max"], size=n)
    im = rng.uniform(d["im_min"], d["im_max"], size=n)
    return re + 1j * im


def tensor_complex(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.complex128, device=device)


def tensor_real(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = choose_device(args.device)

    tcfg = cfg["training"]
    model = ComplexMatrixNet(
        matrix_dim=3,
        hidden_dim=int(tcfg["hidden_dim"]),
        hidden_layers=int(tcfg["hidden_layers"]),
    ).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]))

    history_every = int(tcfg.get("history_every", tcfg["log_every"]))
    history: list[dict[str, float | int]] = []

    for step in range(1, int(tcfg["steps"]) + 1):
        lam = sample_lambda(cfg, int(tcfg["batch_size"]), rng)
        batch = sample_offshell_batch(lam, rng)

        lam_xy = tensor_real(np.column_stack([lam.real, lam.imag]), device)
        eom = tensor_complex(batch.eom, device)
        curvature = tensor_complex(batch.curvature, device)

        q = model(lam_xy)
        q_eom = torch.einsum("bij,bj->bi", q, eom)
        numerator = torch.mean(torch.abs(curvature - q_eom) ** 2)
        denominator = torch.mean(torch.abs(curvature) ** 2) + 1.0e-30
        loss = numerator / denominator

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % history_every == 0 or step == int(tcfg["steps"]):
            history.append({"step": step, "normalized_fqe_loss": float(loss.item())})
        if step == 1 or step % int(tcfg["log_every"]) == 0:
            print(f"step={step:6d} loss={loss.item():.6e}")

    ecfg = cfg["evaluation"]
    n_grid = int(ecfg["grid_size"])
    d = cfg["domain"]
    re = np.linspace(d["re_min"], d["re_max"], n_grid)
    im = np.linspace(d["im_min"], d["im_max"], n_grid)
    rr, ii = np.meshgrid(re, im, indexing="xy")
    lam_grid = (rr + 1j * ii).reshape(-1)

    with torch.no_grad():
        lam_xy = tensor_real(np.column_stack([lam_grid.real, lam_grid.imag]), device)
        q_pred = model(lam_xy).cpu().numpy()

    q_exact = exact_q_matrix(lam_grid)
    q_rel_error = relative_matrix_error(q_pred, q_exact)
    q_offdiag = offdiagonal_fraction(q_pred)
    q_diag_spread = diagonal_spread(q_pred)

    repeats = int(ecfg["samples_per_lambda"])
    lam_eval = np.repeat(lam_grid, repeats)
    eval_batch = sample_offshell_batch(lam_eval, rng)
    q_eval = np.repeat(q_pred, repeats, axis=0)
    predicted_curvature = apply_q(q_eval, eval_batch.eom)
    fqe_rel = relative_residual(eval_batch.curvature, predicted_curvature)

    diagnostic_entries = []
    diagnostic_q_exact = []
    diagnostic_q_learned = []
    for re_lam, im_lam in ecfg["diagnostic_lambdas"]:
        lam = complex(re_lam, im_lam)
        xy = tensor_real(np.array([[lam.real, lam.imag]]), device)
        with torch.no_grad():
            q_here = model(xy).cpu().numpy()[0]
        q_exact_here = exact_q_matrix(np.asarray(lam))
        diagnostic_q_exact.append(q_exact_here)
        diagnostic_q_learned.append(q_here)
        diagnostic_entries.append(
            {
                "lambda": [lam.real, lam.imag],
                "learned_singular_values": singular_values(q_here).tolist(),
                "exact_singular_values": singular_values(q_exact_here).tolist(),
                "learned_operator_norm": float(np.linalg.norm(q_here, ord=2)),
                "exact_operator_norm": float(np.linalg.norm(q_exact_here, ord=2)),
            }
        )

    metrics = {
        "device": str(device),
        "seed": seed,
        "heldout_fqe_relative_residual": fqe_rel,
        "q_relative_matrix_error": q_rel_error,
        "q_offdiagonal_fraction": q_offdiag,
        "q_diagonal_spread": q_diag_spread,
        "exact_stacked_rank_diagnostic_points": stacked_q_rank(np.asarray(diagnostic_q_exact)),
        "learned_stacked_rank_diagnostic_points": stacked_q_rank(np.asarray(diagnostic_q_learned)),
        "diagnostic_points": diagnostic_entries,
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(
        out_dir / "q_grid.npz",
        lam=lam_grid,
        q_pred=q_pred,
        q_exact=q_exact,
        grid_size=n_grid,
    )
    torch.save(model.state_dict(), out_dir / "q_net.pt")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
