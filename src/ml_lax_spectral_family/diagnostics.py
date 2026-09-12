"""Diagnostics for learned F = Q E factorizations."""

from __future__ import annotations

import numpy as np


def relative_residual(lhs: np.ndarray, rhs: np.ndarray, eps: float = 1.0e-30) -> float:
    """Return ||lhs-rhs||_2 / ||lhs||_2 over all entries."""

    num = np.linalg.norm(np.asarray(lhs) - np.asarray(rhs))
    den = np.linalg.norm(np.asarray(lhs)) + eps
    return float(num / den)


def relative_matrix_error(pred: np.ndarray, target: np.ndarray, eps: float = 1.0e-30) -> float:
    """Relative Frobenius error over a batch of matrices."""

    return relative_residual(target, pred, eps=eps)


def offdiagonal_fraction(q: np.ndarray, eps: float = 1.0e-30) -> float:
    """Fraction of Frobenius norm carried by off-diagonal entries."""

    q = np.asarray(q)
    diag = np.zeros_like(q)
    idx = np.arange(q.shape[-1])
    diag[..., idx, idx] = q[..., idx, idx]
    off = q - diag
    return float(np.linalg.norm(off) / (np.linalg.norm(q) + eps))


def diagonal_spread(q: np.ndarray, eps: float = 1.0e-30) -> float:
    """RMS spread of diagonal entries relative to their RMS magnitude."""

    q = np.asarray(q)
    diag = np.diagonal(q, axis1=-2, axis2=-1)
    mean_diag = np.mean(diag, axis=-1, keepdims=True)
    num = np.linalg.norm(diag - mean_diag)
    den = np.linalg.norm(diag) + eps
    return float(num / den)


def singular_values(q: np.ndarray) -> np.ndarray:
    """Return matrix singular values for each sampled spectral parameter."""

    return np.linalg.svd(np.asarray(q), compute_uv=False)
