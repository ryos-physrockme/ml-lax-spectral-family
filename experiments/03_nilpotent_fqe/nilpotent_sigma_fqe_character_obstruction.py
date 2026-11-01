#!/usr/bin/env python3
"""Exact character obstruction to a restricted off-shell F=Q(E) Lax deformation.

Target: five-dimensional two-step nilpotent Lie algebra with
  [e1,e2]=e4, [e1,e3]=t e5,
  standard orthonormal left-invariant metric, and
  H=t*q (theta^124 + theta^135).
At t=0 the target is H3 x R2.

Let U be the reference auxiliary generator for w and suppose there is
an invariant pairing <.,.> such that <U,U> != 0 and U commutes
with every generator in the baseline H3 x R2 Lax connection. Assume
<U, T1>=<U,T2>=<U,T4>=<U,V>=0. The linear functional
chi(A)=<U,A>/<U,U> then annihilates [P0,X] and [X,R0] for EVERY X
by invariance, even if chi is not a character of the full auxiliary Lie
algebra. This includes BOTH traceful and traceless gl(5) embeddings.
It also holds for any compact metric auxiliary Lie algebra containing
the reference direct-sum Lax connection as an orthogonal sector.

Assume a coframe-linear, field-independent Lax connection with a smooth
t-expansion and a reference spectral family at t=0. The coefficient
identities for theta_+^1 theta_-^5 and the reversed monomial, after
applying chi, force q=-1 and q=+1 simultaneously for generic spectral
parameter z != 0, +/-1, independently of any extra generators even when all other auxiliary matrix
directions are allowed. Hence no first-order extension with these
specific assumptions exists. This is NOT a general sigma-model no-go.
"""
from __future__ import annotations
import sympy as s

t,z,q=s.symbols('t z q',real=True)
# MC coefficient f[e1,e3]=t e5; flux components H_135=t q.
# First-order E^3 (= e3 / w) coefficients at monomials (1,5),(5,1):
# k^3_15 = -(1+q)t,  k^3_51=-(1-q)t.
k35_15=-(1+q); k35_51=-(1-q)
# For free reference field w, L_+ = U theta_+^3/(1-z),
# L_- = U theta_-^3/(1+z), and chi(U)=1.
chi_R_minus_P=s.factor(1/(1+z)-1/(1-z))
# With C_ab = [P_a,R_b] -(R-P)_c*k^c_ab/2 -(R+P)_c*f^c_ab/2,
# and no existing f0/k0 in (1,5) or (5,1), the character constraints are:
condition_15=s.factor(-s.Rational(1,2)*chi_R_minus_P*k35_15)
condition_51=s.factor(-s.Rational(1,2)*chi_R_minus_P*k35_51)
assert s.simplify(condition_15+z*(1+q)/(1-z*z))==0
assert s.simplify(condition_51+z*(1-q)/(1-z*z))==0
assert not s.solve([s.together(condition_15/z), s.together(condition_51/z)], [q])
print('Coefficient chi(C_{1,5}) =',condition_15)
print('Coefficient chi(C_{5,1}) =',condition_51)
print('For generic z != 0, +/-1: q = -1 AND q = +1 (incompatible).')
print('Scope: smooth first order, coframe-linear and target-independent Lax;')
print('       an invariant functional paired with a reference commuting generator U.')
print('       Includes traceless or traceful gl(5) realizations at first order.')
print('Does not rule out field-dependent Lax coefficients or alternative baselines.')

