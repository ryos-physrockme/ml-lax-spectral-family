#!/usr/bin/env python3
"""Off-shell F=Q(E) search for a five-dimensional nilpotent sigma model.

Target Lie algebra, in an orthonormal left-invariant coframe theta^a:
    [e1,e2] = e4, [e1,e3] = t e5,
    H = t q (theta^124 + theta^135), G_ab=delta_ab.
At (t,q)=(1,1), this is the harmonic NS flux candidate discussed in
studies of generalized Ricci solitons. At t=0, H3 x R2 is a reference.

E^a = partial_+ theta_-^a + partial_- theta_+^a
      + k^a_{bc} theta_+^b theta_-^c
M^a = partial_+ theta_-^a - partial_- theta_+^a
      + f^a_{bc} theta_+^b theta_-^c = 0 (Maurer--Cartan identity)
L_+ = P_a(z) theta_+^a, L_- = R_a(z) theta_-^a, P_a,R_a in gl(5,R).
F(L) = 1/2 (R-P)_a E^a + 1/2 (R+P)_a M^a
       + C_ab theta_+^a theta_-^b
C_ab = [P_a,R_b] - 1/2 (R-P)_c k^c_ab - 1/2 (R+P)_c f^c_ab.

Therefore C=0 is an *exact off-shell* F=Q(E) certificate (M=0
kinematically), Q_a=(R_a-P_a)/2. Rank(Q)=5 must be tested, since the
trivial flat connection L=0 is otherwise always an optimizer.

The finite-dimensional optimization uses a general gl(5,R) coframe-linear
ansatz at one spectral point. This is a pilot, NOT a proof of a spectral
family, of Poisson-commuting charges, or a non-integrability theorem.

Run with: python nilpotent_sigma_fqe_optimization.py --steps 1000 --seeds 2
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares

N=5

def sign_three(i,j,k):
    if len({i,j,k})!=3: return 0
    indices=[i,j,k]
    return (-1)**sum(indices[a]>indices[b] for a in range(3) for b in range(a+1,3))

def target_coefficients(t:float,q:float):
    """f[c,a,b]=<e^c,[e_a,e_b]>, k from G=I,H=t q(e124+e135)."""
    f=np.zeros((N,N,N),dtype=np.float64)
    for a,b,c,v in [(0,1,3,1.),(0,2,4,float(t))]:
        f[c,a,b]=v;f[c,b,a]=-v
    H=np.zeros((N,N,N),dtype=np.float64)
    for tr in [(0,1,3),(0,2,4)]:
        for i in range(N):
          for j in range(N):
            for k in range(N):
              if sorted([i,j,k])==list(tr):H[i,j,k]=float(t*q)*sign_three(i,j,k)
    # with orthonormal G: k^r_ab = -(f^b_ar + f^a_br) + H_rab
    k=np.empty_like(f)
    for r in range(N):
      for a in range(N):
        for b in range(N):
          k[r,a,b]=-f[b,a,r]-f[a,b,r]+H[r,a,b]
    return f,k

def so3_rotation_matrices():
    # standard so(3) generators satisfy [J1,J2]=J3 cyclically
    def e(i,j):
        v=np.zeros((3,3));v[i,j]=1;return v
    J1=e(2,1)-e(1,2)
    J2=e(0,2)-e(2,0)
    J3=e(1,0)-e(0,1)
    assert np.allclose(J1@J2-J2@J1,J3)
    return J1,J2,J3

def known_reference_lax(z,tracefree=False):
    """Exact H3 x R2 Lax in gl(5) with two central diagonal elements."""
    T1,T2,T4=so3_rotation_matrices()
    matrices=[]
    for T in [T1,T2]:
        M=np.zeros((5,5));M[:3,:3]=T;matrices.append(M)
    U=np.zeros((5,5));U[3,3]=1
    if tracefree: U[4,4]=-1
    matrices.append(U)
    Z=np.zeros((5,5));Z[:3,:3]=T4;matrices.append(Z)
    V=np.zeros((5,5));V[4,4]=1
    if tracefree: V=np.diag([1/3,1/3,1/3,-1/2,-1/2])
    matrices.append(V)
    P=np.stack(matrices)*np.array([1,1,1,-1,1])[:,None,None]/(1-z)
    R=np.stack(matrices)*np.array([-1,-1,1,-1,1])[:,None,None]/(1+z)
    return P,R

def remainder_numpy(P,R,f,k):
    comm=P[:,None]@R[None,:]-R[None,:]@P[:,None]
    return comm - 0.5*np.einsum('cab,cij->abij',k,R-P) -0.5*np.einsum('cab,cij->abij',f,R+P)

def remainder_torch(P,R,f,k):
    comm=P.unsqueeze(1)@R.unsqueeze(0)-R.unsqueeze(0)@P.unsqueeze(1)
    return comm - 0.5*torch.einsum('cab,cij->abij',k,R-P) -0.5*torch.einsum('cab,cij->abij',f,R+P)

def qmap_numpy(P,R):
    return ((R-P)/2).reshape(N,-1).T

def offshell_test(P,R,f,k,seed=10,samples=64):
    """Numerically test F-QE for independent current and EOM jets.

    The Maurer--Cartan jet M is set to zero (kinematic identity).
    Both E and theta_+,theta_- are independently sampled off shell.
    """
    rng=np.random.default_rng(seed)
    Q=(R-P)/2
    res=[]
    c=remainder_numpy(P,R,f,k)
    for _ in range(samples):
        up=rng.normal(size=N); um=rng.normal(size=N);E=rng.normal(size=N)
        Fbil=np.einsum('abij,a,b->ij',c,up,um)
        # actual F from raw derivatives satisfying M=0
        fbil=np.einsum('cab,a,b->c',f,up,um)
        kbil=np.einsum('cab,a,b->c',k,up,um)
        dpum=(E-kbil-fbil)/2
        dmup=(E-kbil+fbil)/2
        F=np.einsum('a,aij->ij',dpum,R)-np.einsum('a,aij->ij',dmup,P)
        F+=np.einsum('a,b,abij->ij',up,um,(P[:,None]@R[None,:]-R[None,:]@P[:,None]))
        QE=np.einsum('a,aij->ij',E,Q)
        assert np.allclose(F-QE,Fbil,rtol=1e-10,atol=1e-10)
        res.append(np.linalg.norm(F-QE))
    return float(np.sqrt(np.mean(np.square(res))))

def score(P,R,f,k,eps=1e-12):
    C=remainder_numpy(P,R,f,k)
    q=qmap_numpy(P,R)
    s=np.linalg.svd(q,compute_uv=False)
    return dict(remainder_rms=float(np.sqrt(np.mean(C**2))),
        remainder_max=float(np.max(abs(C))),q_min_sv=float(s[-1]),
        q_singular_values=[float(x) for x in s],q_rank=float(np.sum(s>1e-6)),
        offshell_rms=offshell_test(P,R,f,k))

def optimize_one(t,q,z,seed,steps,anchor_weight=1e-4,minsv=0.15,rank_weight=0.1,tracefree=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    f_np,k_np=target_coefficients(t,q)
    f=torch.tensor(f_np,dtype=torch.float64); k=torch.tensor(k_np,dtype=torch.float64)
    P0,R0=known_reference_lax(z,tracefree=tracefree)
    p0=torch.tensor(P0,dtype=torch.float64);r0=torch.tensor(R0,dtype=torch.float64)
    # train direct tensors to permit new auxiliary gl(5) directions
    deltaP=torch.nn.Parameter(torch.randn((N,N,N),dtype=torch.float64)*0.02)
    deltaR=torch.nn.Parameter(torch.randn((N,N,N),dtype=torch.float64)*0.02)
    opt=torch.optim.Adam([deltaP,deltaR],lr=0.025)
    best=None
    def evaluate():
        P=p0+deltaP; R=r0+deltaR
        C=remainder_torch(P,R,f,k)
        qcols=((R-P)/2).reshape(N,-1).T
        s=torch.linalg.svdvals(qcols)
        # off-shell curvature coefficient matching. The rank term prevents zero Lax.
        loss1=(C*C).mean()
        loss_rank=(torch.relu(minsv-s)**2).sum()
        loss_anchor=(deltaP.square().mean()+deltaR.square().mean())
        loss=loss1+rank_weight*loss_rank+anchor_weight*loss_anchor
        return loss,loss1,s
    for i in range(steps):
        opt.zero_grad()
        loss,lc,s=evaluate()
        loss.backward()
        opt.step()
        if i%100==0 or i==steps-1:
            val=float(lc.detach())
            if best is None or val<best[0]:
                best=(val,(p0+deltaP).detach().numpy().copy(),(r0+deltaR).detach().numpy().copy())
    # Optional polish with deterministic LBFGS, only when non-trivial branch is found.
    opt2=torch.optim.LBFGS([deltaP,deltaR],lr=0.7,max_iter=100,
                 tolerance_grad=1e-12,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad()
        loss,_,_=evaluate();loss.backward();return loss
    opt2.step(closure)
    P=(p0+deltaP).detach().numpy();R=(r0+deltaR).detach().numpy()
    outcome=score(P,R,f_np,k_np)
    outcome.update(dict(target_t=t,flux_q=q,spectral_z=z,seed=seed,iterations=steps,
                        deviation_from_reference=float(np.sqrt(np.mean((P-P0)**2+(R-R0)**2)))))
    return outcome,P,R

def run(args):
    timestamp=time.time()
    cases=[];weights=[]
    z=args.spectral_z
    for t,q in [(0,0),(1,0),(1,1)]:
        f,k=target_coefficients(t,q)
        Pref,Rref=known_reference_lax(z,tracefree=args.tracefree)
        control=score(Pref,Rref,f,k)
        control.update(target_t=t,flux_q=q,spectral_z=z,baseline=True)
        cases.append(control)
    # Multi-restart optimization on 5d fully interacting target.
    for q in [0,1]:
        for seed in range(args.seeds):
            out,P,R=optimize_one(1,q,z,seed=seed+1,steps=args.steps,
                               anchor_weight=args.anchor_weight,minsv=args.min_singular,rank_weight=args.rank_weight,tracefree=args.tracefree)
            cases.append(out)
            if args.output:
                weights.append(dict(q=q,seed=seed+1,P=P,R=R))
            print('q',q,'seed',seed+1,'rms',round(out['remainder_rms'],7),
                  'minsv',round(out['q_min_sv'],6), 'offshell',round(out['offshell_rms'],6),flush=True)
    if args.output:
       p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
       with open(p,'w',encoding='utf8') as file:
          json.dump(dict(name='5D nilpotent off-shell curvature factorization',
              model='[e1,e2]=e4, [e1,e3]=t e5; G=I; H=t*q(e124+e135)',
              auxiliary='gl(5,R); L± coframe-linear at one generic spectral z',
              convention='F=Q E + R_ab theta+^a theta-^b after Maurer-Cartan',
              cases=cases, runtime_seconds=round(time.time()-timestamp,2)),file,indent=2)
       if weights: np.savez(p.with_suffix('.npz'),**{f'P_q{x["q"]}_seed{x["seed"]}':x['P'] for x in weights},
                                **{f'R_q{x["q"]}_seed{x["seed"]}':x['R'] for x in weights})
    for c in cases[:3]:
       print('CONTROL t=',c['target_t'],'q=',c['flux_q'],'rms=',c['remainder_rms'],
             'minsv=',c['q_min_sv'],'offshell=',c['offshell_rms'])
    return cases

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--spectral-z',type=float,default=0.43)
    parser.add_argument('--anchor-weight',type=float,default=1e-4)
    parser.add_argument('--min-singular',type=float,default=0.15)
    parser.add_argument('--rank-weight',type=float,default=0.1)
    parser.add_argument('--tracefree',action='store_true',help='tracefree embedding of the two commuting reference generators')
    parser.add_argument('--output',type=str,default='results/03_nilpotent_fqe/fixed_spectral_point_results.json')
    run(parser.parse_args())

