import numpy as np

from ml_lax_spectral_family.diagnostics import relative_residual
from ml_lax_spectral_family.pcm import (
    apply_q,
    exact_q_matrix,
    maurer_cartan_residual,
    sample_offshell_batch,
    spectral_curve_gap,
    stacked_q_rank,
    standard_lax_coefficients,
)


def test_standard_pcm_coefficients_lie_on_spectral_curve() -> None:
    lam = np.array([0.0, 0.2, -0.3 + 0.1j, 0.4j], dtype=np.complex128)
    a, c = standard_lax_coefficients(lam)
    np.testing.assert_allclose(spectral_curve_gap(a, c), 0.0, atol=1.0e-13, rtol=1.0e-13)


def test_offshell_sampler_imposes_maurer_cartan_only() -> None:
    rng = np.random.default_rng(7)
    lam = np.full(128, 0.3 + 0.2j, dtype=np.complex128)
    batch = sample_offshell_batch(lam, rng)
    np.testing.assert_allclose(maurer_cartan_residual(batch), 0.0, atol=1.0e-13, rtol=0.0)
    assert np.linalg.norm(batch.eom) > 1.0


def test_exact_f_equals_q_e_off_shell() -> None:
    rng = np.random.default_rng(11)
    lam = rng.uniform(-0.7, 0.7, 512) + 1j * rng.uniform(-0.7, 0.7, 512)
    batch = sample_offshell_batch(lam, rng)
    q = exact_q_matrix(lam)
    q_e = apply_q(q, batch.eom)
    assert relative_residual(batch.curvature, q_e) < 1.0e-12


def test_rank_diagnostic_has_known_degenerate_point() -> None:
    q_zero = exact_q_matrix(np.asarray(0.0 + 0.0j))
    q_generic = exact_q_matrix(np.asarray(0.25 + 0.15j))
    assert stacked_q_rank(np.asarray([q_zero])) == 0
    assert stacked_q_rank(np.asarray([q_generic])) == 3
    assert stacked_q_rank(np.asarray([q_zero, q_generic])) == 3
