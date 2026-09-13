import numpy as np

from ml_lax_spectral_family.s2 import (
    apply_q,
    exact_q_matrix,
    sample_offshell_batch,
)


def test_s2_exact_factorization_on_offshell_samples():
    rng = np.random.default_rng(123)
    lam = np.array([0.7 + 0.2j, 1.0 + 0.0j, 1.3 - 0.1j, 0.8 - 0.3j])
    batch = sample_offshell_batch(lam, rng)
    predicted = apply_q(exact_q_matrix(lam), batch.eom)
    np.testing.assert_allclose(batch.curvature, predicted, atol=1.0e-12, rtol=1.0e-12)


def test_s2_q_vanishes_at_lambda_one():
    q = exact_q_matrix(np.asarray(1.0 + 0.0j))
    np.testing.assert_allclose(q, 0.0, atol=1.0e-14)


def test_s2_q_has_rank_two_at_generic_lambda():
    q = exact_q_matrix(np.asarray(0.8 + 0.2j))
    singular_values = np.linalg.svd(q, compute_uv=False)
    assert np.count_nonzero(singular_values > 1.0e-12) == 2
