"""S^2 = SU(2)/U(1) benchmark for the off-shell relation F = Q E.

The coordinate fields are theta and phi.  The two independent equation-of-
motion residuals used here are

    E_theta = theta_{+-} - sin(theta) cos(theta) phi_+ phi_-,

and

    E_phi = sin(theta) phi_{+-}
            + cos(theta) (theta_+ phi_- + theta_- phi_+).

For the standard symmetric-coset Lax connection

    L_+ = J_+^(0) + lambda J_+^(1),
    L_- = J_-^(0) + lambda^(-1) J_-^(1),

the curvature components in a basis (T1,T2,T3) satisfy

    F1 = (lambda^(-1) - lambda) E_phi,
    F2 = (lambda - lambda^(-1)) E_theta,
    F3 = 0.

The sampling routine constructs coordinate jets with arbitrary nonzero
E_theta and E_phi, so it is off shell with respect to the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S2OffShellBatch:
    lam: np.ndarray
    theta: np.ndarray
    phi_plus: np.ndarray
    phi_minus: np.ndarray
    theta_plus: np.ndarray
    theta_minus: np.ndarray
    phi_plus_minus: np.ndarray
    theta_plus_minus: np.ndarray
    e_phi: np.ndarray
    e_theta: np.ndarray
    curvature: np.ndarray

    @property
    def eom(self) -> np.ndarray:
        """Return residual vector ordered as (E_phi, E_theta)."""

        return np.column_stack([self.e_phi, self.e_theta])


def standard_lax_coefficients(lam: np.ndarray | complex) -> tuple[np.ndarray, np.ndarray]:
    """Return b(lambda)=lambda and d(lambda)=1/lambda."""

    lam_arr = np.asarray(lam, dtype=np.complex128)
    if np.any(np.abs(lam_arr) == 0.0):
        raise ValueError("lambda = 0 is a pole of the chosen S^2 parametrization")
    return lam_arr, 1.0 / lam_arr


def exact_q_matrix(lam: np.ndarray | complex) -> np.ndarray:
    """Return the exact 3 x 2 map from (E_phi,E_theta) to curvature."""

    lam_arr = np.asarray(lam, dtype=np.complex128)
    b, d = standard_lax_coefficients(lam_arr)
    q = np.zeros(lam_arr.shape + (3, 2), dtype=np.complex128)
    q[..., 0, 0] = d - b
    q[..., 1, 1] = b - d
    return q


def curvature_from_coordinate_jets(
    lam: np.ndarray,
    theta: np.ndarray,
    phi_plus: np.ndarray,
    phi_minus: np.ndarray,
    theta_plus: np.ndarray,
    theta_minus: np.ndarray,
    phi_plus_minus: np.ndarray,
    theta_plus_minus: np.ndarray,
) -> np.ndarray:
    """Evaluate the three curvature components before imposing the equations of motion."""

    lam = np.asarray(lam, dtype=np.complex128)
    b, d = standard_lax_coefficients(lam)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    f1 = (
        (d - b) * sin_theta * phi_plus_minus
        + cos_theta
        * (
            (d - b) * phi_minus * theta_plus
            + (d - b) * phi_plus * theta_minus
        )
    )
    f2 = (
        (b - d) * theta_plus_minus
        + (d - b) * sin_theta * cos_theta * phi_plus * phi_minus
    )

    bd = b * d
    f3 = sin_theta * (
        (bd - 1.0) * phi_minus * theta_plus
        + (1.0 - bd) * phi_plus * theta_minus
    )
    return np.column_stack([f1, f2, f3])


def sample_offshell_batch(
    lam: np.ndarray,
    rng: np.random.Generator,
    derivative_scale: float = 1.0,
    eom_scale: float = 1.0,
    theta_margin: float = 0.35,
) -> S2OffShellBatch:
    """Generate coordinate jets with independently sampled equation-of-motion residuals."""

    lam = np.asarray(lam, dtype=np.complex128).reshape(-1)
    if np.any(np.abs(lam) == 0.0):
        raise ValueError("lambda = 0 is excluded from this parametrization")
    n = lam.shape[0]

    theta = rng.uniform(theta_margin, np.pi - theta_margin, size=n)
    phi_plus = rng.normal(scale=derivative_scale, size=n)
    phi_minus = rng.normal(scale=derivative_scale, size=n)
    theta_plus = rng.normal(scale=derivative_scale, size=n)
    theta_minus = rng.normal(scale=derivative_scale, size=n)
    e_phi = rng.normal(scale=eom_scale, size=n)
    e_theta = rng.normal(scale=eom_scale, size=n)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_plus_minus = e_theta + sin_theta * cos_theta * phi_plus * phi_minus
    phi_plus_minus = (
        e_phi
        - cos_theta * (theta_plus * phi_minus + theta_minus * phi_plus)
    ) / sin_theta

    curvature = curvature_from_coordinate_jets(
        lam=lam,
        theta=theta,
        phi_plus=phi_plus,
        phi_minus=phi_minus,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        phi_plus_minus=phi_plus_minus,
        theta_plus_minus=theta_plus_minus,
    )

    return S2OffShellBatch(
        lam=lam,
        theta=theta,
        phi_plus=phi_plus,
        phi_minus=phi_minus,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        phi_plus_minus=phi_plus_minus,
        theta_plus_minus=theta_plus_minus,
        e_phi=e_phi,
        e_theta=e_theta,
        curvature=curvature,
    )


def apply_q(q: np.ndarray, eom: np.ndarray) -> np.ndarray:
    """Apply a batch of complex 3 x 2 matrices to residual vectors."""

    return np.einsum("...ij,...j->...i", q, eom)
