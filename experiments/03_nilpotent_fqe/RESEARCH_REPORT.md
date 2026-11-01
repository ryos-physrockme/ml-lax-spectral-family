# Off-shell Lax-factorization search for a five-dimensional nilpotent sigma model

**Date:** 2026-11-01  
**Status:** exploratory numerical search and one restricted analytic obstruction; no integrability claim.

## 1. Precisely specified model

Let the simply connected five-dimensional two-step nilpotent Lie group have Lie algebra

\[
[e_1,e_2]=e_4,\qquad [e_1,e_3]=e_5,
\]

with orthonormal left-invariant one-forms

\[
\theta^1=dx,\quad \theta^2=dy,\quad\theta^3=dw,\quad
\theta^4=dz_1-x\,dy,\quad\theta^5=dz_2-x\,dw.
\]

The worldsheet is Lorentzian, with two light-cone coordinates. The target-space metric and closed flux are

\[
g=\sum_{a=1}^5 \theta^a\otimes\theta^a,\qquad
H=q\,(\theta^{124}+\theta^{135}).
\]

The two concrete targets studied are \(q=0\) (no NS flux) and \(q=1\) (closed, co-closed nonzero NS flux). The model at \(q=1\) provides a generalized Ricci soliton example. This property does not itself imply integrability. For interpolation from the reference \(H_3\times\mathbb R^2\), use \([e_1,e_3]=t e_5\) and \(H=tq(\theta_t^{124}+\theta_t^{135})\), so the reference at \(t=0\) has no flux.

## 2. Off-shell factorization, not on-shell-only fitting

Define, in the left-invariant frame,

\[
\begin{aligned}
M^c&=\partial_+\theta_-^c-\partial_-\theta_+^c+f^c{}_{ab}\theta_+^a\theta_-^b=0,\\
E^c&=\partial_+\theta_-^c+\partial_-\theta_+^c+k^c{}_{ab}\theta_+^a\theta_-^b,
\end{aligned}
\]

where \(M=0\) is the kinematic Maurer–Cartan identity and

\[
k^r{}_{ab}=-f^b{}_{ar}-f^a{}_{br}+H_{rab}.
\]

For a Lax ansatz \(L_+=P_a(Y;z)\theta_+^a\), \(L_-=R_a(Y;z)\theta_-^a\), its off-shell curvature is

\[
F_{+-}=Q_cE^c+\tfrac12(R_c+P_c)M^c+C_{ab}\theta_+^a\theta_-^b,
\qquad Q_c=\tfrac12(R_c-P_c).
\]

For target-coordinate-independent coefficients,

\[
C_{ab}=[P_a,R_b]-\tfrac12(R_c-P_c)k^c{}_{ab}
-\tfrac12(R_c+P_c)f^c{}_{ab}.
\]

For target-coordinate-dependent coefficients, add \(e_aR_b-e_bP_a\) to \(C_{ab}\). The optimization minimizes the complete polynomial coefficient residual \(C_{ab}\), *not only samples with \(E=0\)*. Independent off-shell jets are used for validation, imposing only \(M=0\).

Since the coefficient of the second derivatives fixes \(Q=(R-P)/2\) uniquely within this ansatz, no independent unconstrained neural network for \(Q\) is needed. A full-rank condition \(\operatorname{rank}Q=5\) at generic spectral values is essential; numerical experiments monitor its smallest singular value and penalize rank loss.

## 3. Analytic first-order obstruction (restricted but exact)

At the reference \(t=0\), take the known \(H_3\) Lax connection valued in \(\mathfrak{su}(2)\), plus two abelian currents for the free scalars. Suppose \(U\) denotes the auxiliary generator for \(w\), commuting with the entire reference connection, with a nondegenerate invariant pairing allowing projection onto \(U\).

For a smooth first-order deformation which remains coframe-linear with **target-coordinate-independent** coefficients, the projected curvature-remainder coefficients of \(\theta_+^1\theta_-^5\) and \(\theta_+^5\theta_-^1\), up to an overall sign convention, impose

\[
\frac{z(1+q)}{1-z^2}=0,\qquad
\frac{z(1-q)}{1-z^2}=0.
\]

For generic \(z\ne0,\pm1\), the equations are incompatible. This remains true when the auxiliary connection takes values in a larger matrix Lie algebra \(\mathfrak{gl}(5,\mathbb R)\) and the two free-field generators are tracefree: first-order commutators vanish after projection with the invariant trace pairing and a commuting reference generator.

This **does not rule out** coordinate-dependent Lax coefficients, another reference embedding, discontinuous/nonperturbative branches, or a non-coframe-linear/nonlocal ansatz. It is not a proof of non-integrability of the five-dimensional sigma model.

The numerical linearized \(\mathfrak{gl}(5)\) systems at \(z=1/2\) give 625 coefficient equations for 250 unknown first-order Lax coefficients. The stable floating-point rank is 206. For \(q=0,1,-1\), all have nonzero least-squares residual; the exact projected contradiction provides the analytical justification under the stated assumptions.

## 4. Numerical optimization and controls

All numbers below are *RMS of the 625 coefficient entries* \(C_{ab}\) (5 physical indices on each current, and 25 entries of each \(5\times5\) auxiliary matrix), not numerical tests of spectral integrability.

| Experiment | Flux coefficient | Auxiliary ansatz | RMS remainder | Smallest singular value of Q |
|---|---:|---|---:|---:|
| Exact reference at t=0 (positive control) | 0 | embedded su(2) + R² in gl(5) | below 1e-14 | 0.482 (tracefree embedding, z=0.43) |
| t=1, best 3-seed run, fixed z=0.43 | 0 | constant coframe coefficients, gl(5), tracefree reference | 0.00227 | 0.450 |
| t=1, best 3-seed run, fixed z=0.43 | 1 | same | 0.00430 | 0.450 |
| t=1, spectral rational numerator quadratic, five held-out z values | 0 | constant coframe coefficients, gl(5) | 0.00325 | 0.399 |
| t=1, spectral rational numerator quadratic, five held-out z values | 1 | same | 0.00456 | 0.399 |
| t=1, x-linear, held-out x values, single z=0.43 | 0 | P_a(x)=P_a^(0)+xP_a^(1), similarly R_a | 0.00112 | 0.389 |
| t=1, x-linear, held-out x values, single z=0.43 | 1 | same | 0.00290 | 0.398 |

**Interpretation:** For the chosen finite families, no exact \(F=Q(E)\) certificate has emerged. The coordinate-dependent ansatz improves the numerical residual at a fixed spectral point, but this is not a spectral-parameter family and is not yet an integrability result. Several optimized smallest singular values sit near the imposed penalty threshold, indicating that attempts to diminish Q remain a significant optimization hazard. Numerical smallness is not a theorem of existence or nonexistence.

For the rational spectral search,

\[
P_a(z)=\frac{A_{a0}+zA_{a1}+z^2A_{a2}}{1-z},\qquad
R_a(z)=\frac{B_{a0}+zB_{a1}+z^2B_{a2}}{1+z}.
\]

Six training and five held-out real spectral values were used, away from poles. Polynomial consistency still requires symbolic verification of \(C_{ab}(z)\equiv0\). The coordinate-dependent run uses a single \(z\) and is a separate initial diagnostic; comparisons of residual values across ansätze are exploratory.

## 5. Reproduction

Requires Python 3.11+, NumPy, SciPy, SymPy, PyTorch, pytest. From the directory containing the scripts:

```bash
python nilpotent_sigma_fqe_character_obstruction.py
python nilpotent_sigma_fqe_linearized_gl5.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nilpotent_sigma_fqe_optimization.py --steps 2200 --seeds 3 --tracefree --min-singular 0.45 --rank-weight 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nilpotent_sigma_fqe_spectral.py --steps 1800 --seeds 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python nilpotent_sigma_fqe_coordinate_dependent.py --steps 800 --seeds 1
pytest -q test_nilpotent_sigma_fqe.py
```

Use `--tracefree` for the spectral scan to reproduce its tracefree-reference results, and `--steps 3000 --seeds 1` for the reported tracefree spectral run. Optimizer configurations and pseudorandom seeds are specified in the source. Full numerical results are included in accompanying JSON files.

## 6. Interpretation and next steps

1. The next meaningful extension is a nontrivial *spectral family with coordinate-dependent coefficients*, with a gauge-fixing prescription to distinguish genuinely new connections from gauge-transformed reference Lax pairs.
2. Solve or symbolically certify the polynomial off-shell identities before claiming a Lax representation. A nonsingular Q at at least one generic z is required, together with a non-gauge-trivial spectral parameter. Ultimately check Poisson-commuting conserved charges.
3. Compare to known constructions (T-duality, Poisson–Lie duality, WZW/coset and generalized complex-structure-based models) at each milestone. In particular no non-integrability claim is warranted from a finite ansatz failing to converge.

## References

- B. E. Baaquie and K. K. Yim, *Sigma Model Lagrangian for the Heisenberg Group*, Phys. Lett. B 615 (2005), [arXiv:hep-th/0203156](https://arxiv.org/abs/hep-th/0203156).
- N. Mohammedi, *Classical Integrability of Two Dimensional Non Linear Sigma Models*, Phys. Lett. B 551 (2003), [arXiv:hep-th/0206179](https://arxiv.org/abs/hep-th/0206179).
- N. Mohammedi, *On the geometry of classically integrable two-dimensional non-linear sigma models*, Nucl. Phys. B (2010), [arXiv:0806.0550](https://arxiv.org/abs/0806.0550).
- S. Krippendorf, D. Lüst and M. Syväri, *Integrability ex machina*, Fortschr. Phys. 69 (2021), [arXiv:2103.07475](https://arxiv.org/abs/2103.07475).
- A. Rezaei-Aghdam and A. Taghavi, *Integrable sigma models with complex and generalized complex structures*, [arXiv:2309.13413](https://arxiv.org/abs/2309.13413).