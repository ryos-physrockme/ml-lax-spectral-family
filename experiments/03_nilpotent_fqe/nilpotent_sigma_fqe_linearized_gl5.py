#!/usr/bin/env python3
"""Linearized off-shell F=Q(E) obstruction for a gl(5,R) reference Lax pair.

Coframe linear ansatz, target bracket [e1,e2]=e4, [e1,e3]=t e5,
standard metric, H=t q(e124+e135). At t=0 use known H3 x R2 Lax.
This script solves for all 250 unconstrained first-order corrections to
P_a, R_a in the full gl(5,R) matrix auxiliary algebra at z=1/2.
Matrix ranks are numerical. The separate character_obstruction.py script
provides an exact analytic contradiction for an important baseline sector.
"""
from __future__ import annotations
import json, argparse
import numpy as np
from scipy.linalg import lstsq
import nilpotent_sigma_fqe_optimization as base

N=5

def first_order_matrix(z,q,tracefree=False):
    f0,k0=base.target_coefficients(0,0)
    ffull,kfull=base.target_coefficients(1,q)
    f1=ffull-f0;k1=kfull-k0
    P0,R0=base.known_reference_lax(z,tracefree=tracefree)
    def lin(P1,R1):
        # curvature of first-order Lax correction plus kinematic terms
        comm=(P1[:,None]@R0[None,:]-R0[None,:]@P1[:,None]
             +P0[:,None]@R1[None,:]-R1[None,:]@P0[:,None])
        return (comm-0.5*np.einsum('cab,cij->abij',k0,R1-P1)
                   -0.5*np.einsum('cab,cij->abij',f0,R1+P1))
    source=(0.5*np.einsum('cab,cij->abij',k1,R0-P0)
            +0.5*np.einsum('cab,cij->abij',f1,R0+P0))
    A=np.zeros((N**4,2*N**3),dtype=np.float64)
    for i in range(2*N**3):
        D=np.zeros((N,N,N),dtype=np.float64)
        D.reshape(-1)[i%(N**3)]=1.
        C=lin(D,np.zeros_like(D)) if i<N**3 else lin(np.zeros_like(D),D)
        A[:,i]=C.reshape(-1)
    rhs=source.reshape(-1)
    sol,_,rank,s=lstsq(A,rhs,cond=1e-10,lapack_driver='gelsd')
    residual=A@sol-rhs
    return dict(z=z,q=q,tracefree=tracefree,rank=int(rank),unknowns=A.shape[1],
                equations=A.shape[0],l2_residual=float(np.linalg.norm(residual)),
                max_residual=float(np.max(abs(residual))),rhs_norm=float(np.linalg.norm(rhs)),
                singular_min_nonzero=float(s[rank-1]),
                relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(rhs)))

def main():
    results=[]
    for tracefree in [False,True]:
        for q in [0.,1.,-1.]:
           r=first_order_matrix(z=0.5,q=q,tracefree=tracefree)
           results.append(r)
           print('tracefree',tracefree,'q',q,'rank',r['rank'],'relres',r['relative_residual'],flush=True)
    from pathlib import Path
    outfile=Path('results/03_nilpotent_fqe/linearized_gl5_results.json')
    outfile.parent.mkdir(parents=True,exist_ok=True)
    with open(outfile,'w') as f:json.dump({'description':'First-order coefficient residuals for a gl5 auxiliary algebra','cases':results},f,indent=2)
if __name__=='__main__':main()

