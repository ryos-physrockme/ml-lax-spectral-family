#!/usr/bin/env python3
"""Spectral-parameter off-shell F=Q(E) scan with a gl(5,R) auxiliary algebra.

Uses the self-contained definitions in nilpotent_sigma_fqe_optimization.py.
For each lambda, L+ = P(lambda)_a theta_+^a, L- = R(lambda)_a theta_-^a,

 P(lambda) = sum_{m=0}^2 lambda^m A_m / (1-lambda)
 R(lambda) = sum_{m=0}^2 lambda^m B_m / (1+lambda)

A_m, B_m are unrestricted real 5x5 matrices for each physical component.
All off-shell coefficient identities and Q rank checks are enforced simultaneously
at several spectral samples; withheld spectral samples assess generalization.

This is a finite ansatz and numeric diagnostic, not proof of non-integrability.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import nilpotent_sigma_fqe_optimization as base

DTYPE=torch.float64

def to_predict(A,B,zs):
    zp=torch.stack([zs**j for j in range(A.shape[0])],dim=-1) # nz,nterm
    P=torch.einsum('zm,mabc->zabc',zp,A)/(1-zs[:,None,None,None])
    R=torch.einsum('zm,mabc->zabc',zp,B)/(1+zs[:,None,None,None])
    return P,R

def get_baseline(tracefree=False):
    P0,R0=base.known_reference_lax(0,tracefree=tracefree)
    AA=torch.zeros((3,5,5,5),dtype=DTYPE)
    BB=torch.zeros((3,5,5,5),dtype=DTYPE)
    AA[0]=torch.as_tensor(P0,dtype=DTYPE)
    BB[0]=torch.as_tensor(R0,dtype=DTYPE)
    return AA,BB

def fit(q,steps,seed,train_z,min_sv=0.4,rank_weight=2.,l2weight=3e-6,tracefree=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    A0,B0=get_baseline(tracefree=tracefree)
    A=torch.nn.Parameter(A0+torch.randn_like(A0)*0.01)
    B=torch.nn.Parameter(B0+torch.randn_like(B0)*0.01)
    f,k=base.target_coefficients(t=1.,q=q)
    f=torch.tensor(f,dtype=DTYPE); k=torch.tensor(k,dtype=DTYPE)
    z=torch.tensor(train_z,dtype=DTYPE)
    opt=torch.optim.Adam([A,B],lr=0.018)
    def residual():
        P,R=to_predict(A,B,z)
        comm=P[:,:,None]@R[:,None,:]-R[:,None,:]@P[:,:,None]
        C=comm-0.5*torch.einsum('cab,zcij->zabij',k,R-P)-0.5*torch.einsum('cab,zcij->zabij',f,R+P)
        qmat=((R-P)/2).reshape(len(z),5,-1).transpose(1,2)
        svals=torch.linalg.svdvals(qmat)
        err=(C*C).mean()
        rankerr=(torch.relu(min_sv-svals)**2).mean()
        reg=(A-A0).square().mean()+(B-B0).square().mean()
        loss=err+rank_weight*rankerr+l2weight*reg
        return loss,err,rankerr,svals
    best=None
    for i in range(steps):
        opt.zero_grad()
        loss,err,_,_=residual()
        loss.backward();opt.step()
        if i%100==0 or i==steps-1:
            ev=float(err.detach())
            if best is None or ev<best[0]: best=(ev,A.detach().clone(),B.detach().clone())
    opt2=torch.optim.LBFGS([A,B],lr=0.5,max_iter=80,tolerance_grad=1e-12,line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad()
        loss,_,_,_=residual(); loss.backward()
        return loss
    opt2.step(closure)
    return A.detach().numpy(),B.detach().numpy()

def evaluate(A,B,q,zs):
    z=torch.tensor(zs,dtype=DTYPE)
    with torch.no_grad():
        Ps,Rs=to_predict(torch.tensor(A),torch.tensor(B),z)
    f,k=base.target_coefficients(t=1.,q=q)
    out=[]
    for i,lam in enumerate(zs):
        P,R=Ps[i].numpy(),Rs[i].numpy()
        x=base.score(P,R,f,k)
        out.append(dict(z=float(lam),rem=float(x['remainder_rms']),
                        offshell=float(x['offshell_rms']),
                        min_sv=float(x['q_min_sv'])))
    return out

def main(args):
    train_z=[-.65,-.43,-.16,.24,.47,.68]
    held_z=[-.54,-.31,.11,.38,.58]
    cases=[]
    for q in [0.,1.]:
       for seed in range(1,args.seeds+1):
          A,B=fit(q,args.steps,seed,train_z,min_sv=args.min_singular,
                   rank_weight=args.rank_weight,l2weight=args.anchor_weight,tracefree=args.tracefree)
          train=evaluate(A,B,q,train_z)
          held=evaluate(A,B,q,held_z)
          summary=dict(flux=q,seed=seed,steps=args.steps,
              train_rms=float(np.mean([o['rem'] for o in train])),
              holdout_rms=float(np.mean([o['rem'] for o in held])),
              holdout_min_singular=float(min(o['min_sv'] for o in held)),
              train=train,heldout=held)
          print('q',q,'seed',seed,'train_rms',round(summary['train_rms'],7),
                'held_rms',round(summary['holdout_rms'],7),
                'min Q sv',round(summary['holdout_min_singular'],5),flush=True)
          cases.append(summary)
          if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            np.savez(Path(args.output).with_name('nilpotent_sigma_fqe_spectral_weights_'+('traceless_' if args.tracefree else 'traceful_')+'q'+str(int(q))+'_seed'+str(seed)+'.npz'),A=A,B=B)
    if args.output:
        with open(args.output,'w') as file:
            json.dump(dict(description='Global rational spectral-parameter fit for off-shell F=Q(E)',
                     target='G=I, [e1,e2]=e4,[e1,e3]=e5,H=q(e124+e135)',
                     auxiliary='gl(5,R)',tracefree_reference=args.tracefree,basis='polynomial quadratic numerator over 1∓lambda',
                     train_z=train_z,heldout_z=held_z,cases=cases),file,indent=2)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--min-singular',type=float,default=0.4)
    parser.add_argument('--rank-weight',type=float,default=2.)
    parser.add_argument('--tracefree',action='store_true')
    parser.add_argument('--anchor-weight',type=float,default=3e-6)
    parser.add_argument('--output',default='results/03_nilpotent_fqe/rational_spectral_results.json')
    main(parser.parse_args())

