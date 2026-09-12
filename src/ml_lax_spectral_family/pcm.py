"""Principal-chiral-model benchmark for the off-shell relation F = Q E.

Conventions
-----------
The Lie algebra basis satisfies [T_a, T_b] = epsilon_{abc} T_c, so the
commutator coefficients are represented by the ordinary three-dimensional
cross product. The Maurer--Cartan residual and equation-of-motion residual
are

    M = d_+ J_- - d_- J_+ + [J_+, J_-],
    E = d_+ J_- + d_- J_+.

Off-shell samples in this module satisfy M = 0 exactly but do not impose
E = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCMOffShellBatch:
    """A batch of PCM jets satisfying the Maurer--Cartan identity."""

    lam: np.ndarray
    j_plus: np.ndarray
    j_minus: np.ndarray
    eom: np.ndarray
    d_plus_j_minus: np.ndarray
    d_minus_j_plus: np.ndarray
    bracket: np.ndarray
    curvature: np.ndarray


def standard_lax_coefficients(lam: np.ndarray | complex) -> tuple[np.ndarray, np.ndarray]:
    """Return a(lambda), c(lambda) for the standard PCM Lax connection."""

    lam_arr = np.asarray(lam, dtype=np.complex128)
    return 1.0 / (1.0 - lam_arr), 1.0 / (1.0 + lam_arr)


def spectral_curve_gap(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return a + c - 2 a c."""

    return a + c - 2.0 * a * c


def exact_q_scalar(lam: np.ndarray | complex) -> np.ndarray:
    """Exact scalar multiplying the identity in Q(lambda)."""

    lam_arr = np.asarray(lam, dtype=np.complex128)
    return -lam_arr / (1.0 - lam_arr**2)


def exact_q_matrix(lam: np.ndarray | complex) -> np.ndarray:
    """Return Q(lambda) = q(lambda) I_3 as a complex array."""

    q = exact_q_scalar(lam)
    eye = np.eye(3, dtype=np.complex128)
    return q[..., None, None] * eye


def curvature_from_jets(
    a: np.ndarray,
    c: np.ndarray,
    d_plus_j_minus: np.ndarray,
    d_minus_j_plus: np.ndarray,
    bracket: np.ndarray,
) -> np.ndarray:
    """Evaluate F_{+-} for L_+ = a J_+, L_- = c J_-."""

    return (
        c[..., None] * d_plus_j_minus
        - a[..., None] * d_minus_j_plus
        + (a * c)[..., None] * bracket
    )


def sample_offshell_batch(
    lam: np.ndarray,
    rng: np.random.Generator,
    field_scale: float = 1.0,
    eom_scale: float = 1.0,
) -> PCMOffShellBatch:
    """Generate off-shell PCM samples with Maurer--Cartan imposed exactly.

    J_+, J_- and E are sampled independently. Writing C = [J_+, J_-],
    the two derivative jets are chosen as

        d_+ J_- = (E - C)/2,
        d_- J_+ = (E + C)/2,

    which gives M = 0 while leaving E arbitrary.
    """

    lam = np.asarray(lam, dtype=np.complex128).reshape(-1)
    n = lam.shape[0]
    j_plus = rng.normal(scale=field_scale, size=(n, 3))
    j_minus = rng.normal(scale=field_scale, size=(n, 3))
    eom = rng.normal(scale=eom_scale, size=(n, 3))
    bracket = np.cross(j_plus, j_minus)
    d_plus_j_minus = 0.5 * (eom - bracket)
    d_minus_j_plus = 0.5 * (eom + bracket)
    a, c = standard_lax_coefficients(lam)
    curvature = curvature_from_jets(a, c, d_plus_j_minus, d_minus_j_plus, bracket)
    return PCMOffShellBatch(
        lam=lam,
        j_plus=j_plus,
        j_minus=j_minus,
        eom=eom,
        d_plus_j_minus=d_plus_j_minus,
        d_minus_j_plus=d_minus_j_plus,
        bracket=bracket,
        curvature=curvature,
    )


def maurer_cartan_residual(batch: PCMOffShellBatch) -> np.ndarray:
    """Evaluate M on a generated batch; this should vanish to roundoff."""

    return batch.d_plus_j_minus - batch.d_minus_j_plus + batch.bracket


def apply_q(q: np.ndarray, eom: np.ndarray) -> np.ndarray:
    """Apply a batch of complex 3x3 Q matrices to real/complex E vectors."""

    return np.einsum("...ij,...j->...i", q, eom)


def stacked_q_rank(q_matrices: np.ndarray, atol: float = 1.0e-10) -> int:
    """Rank of vertically stacked Q(lambda_i) matrices.

    For several spectral parameters this tests whether the combined curvature
    equations constrain every equation-of-motion direction.
    """

    q_matrices = np.asarray(q_matrices, dtype=np.complex128)
    stacked = q_matrices.reshape(-1, q_matrices.shape[-1])
    singular_values = np.linalg.svd(stacked, compute_uv=False)
    return int(np.count_nonzero(singular_values > atol))
