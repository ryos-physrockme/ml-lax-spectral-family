# Experiment 01 multi-seed summary

Independent runs: 5
Seeds: 1234, 2026, 2718, 4242, 31415

| diagnostic | mean | standard deviation | minimum | maximum |
| --- | ---: | ---: | ---: | ---: |
| heldout_fqe_relative_residual | 2.964360e-02 | 5.569931e-03 | 2.096773e-02 | 3.445464e-02 |
| q_relative_matrix_error | 2.967890e-02 | 5.590462e-03 | 2.096671e-02 | 3.456865e-02 |
| q_offdiagonal_fraction | 1.246096e-02 | 6.198763e-03 | 6.478096e-03 | 2.234754e-02 |
| q_diagonal_spread | 4.373163e-03 | 6.795124e-04 | 3.505866e-03 | 5.349938e-03 |

At spectral parameter lambda = 0 the exact map Q vanishes. The learned operator norm therefore measures approximation error rather than a nonzero rank:

- mean learned operator norm: 1.628056e-02
- standard deviation: 5.118534e-03

Generated figures:

- global_metrics_by_seed.png
- q_diagonal_real_axis_multiseed.png
- q_error_complex_plane_multiseed_mean.png
- q_singular_values_lambda_zero_by_seed.png
