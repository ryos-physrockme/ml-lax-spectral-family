# Experiment 01 numerical summary

The neural network was trained to learn a general complex 3 x 3 matrix Q(lambda) from off-shell principal-chiral-model samples satisfying the Maurer--Cartan identity.
The analytic form of Q(lambda) was used only after training for validation.

- Held-out relative residual ||F-QE||/||F||: 2.830418e-02
- Relative matrix error for Q: 2.829686e-02
- Fraction of matrix norm in off-diagonal entries: 7.491358e-03
- Relative spread among diagonal entries: 4.886591e-03
- Rank of vertically stacked exact Q matrices at diagnostic points: 3
- Rank of vertically stacked learned Q matrices at diagnostic points: 3

Generated figures:

- training_loss.png
- q_diagonal_real_axis.png
- q_offdiagonal_real_axis.png
- q_error_complex_plane.png
- q_singular_values_real_axis.png
