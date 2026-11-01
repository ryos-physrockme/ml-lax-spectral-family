#!/usr/bin/env python3
"""Off-shell F=Q(E) scan with coordinate-dependent coframe-linear Lax coefficients.

Target group N5: theta=(dx,dy,dw,dz1-xdy,dz2-xdw),
[e1,e2]=e4,[e1,e3]=e5, metric g=sum(theta^a)^2,
H=q(theta^124+theta^135).

A single real spectral parameter z=0.43 is tested at this stage.
The auxiliary algebra is gl(5,R). The ansatz adds x dependence:
   L_+ = (P_a^(0) + x P_a^(1)) theta_+^a
   L_- = (R_a^(0) + x R_a^(1)) theta_-^a.

The exact off-shell F-QE remainder in the left-invariant basis is
 C_ab(x)=[P_a(x),R_b(x)] + delta_{a1} partial_x R_b(x)
        -delta_{b1} partial_x P_a(x)
        -(R-P)_c k^c_ab/2 -(R+P)_c f^c_ab/2.
This holds because e1=partial_x and e_{2,3,4,5}x=0.
Q_a(x)=(R_a(x)-P_a(x))/2.

A small remainder at finitely many x values is NOT a certificate of
integrability; inspect held-out x, Q rank, spectral continuation,
possible gauge triviality, and a symbolic off-shell identity.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import nilpotent_sigma_fqe_optimization as base

DT=torch.float64

def coefficient_remainder(P0,R0,P1,R1,x,f,k):
    P=P0+x*P1;R=R0+x*R1
    comm=P[:,None]@R[None,:]-R[None,:]@P[:,None]
    derivative=torch.zeros_like(comm)
    derivative[0,:,:,:]+=R1
    derivative[:,0,:,:]-=P1
    C=comm+derivative-0.5*torch.einsum('cab,cij->abij',k,R-P)\
                     -0.5*torch.einsum('cab,cij->abij',f,R+P)
    Qmat=((R-P)/2).reshape(5,-1).T
    return C,Qmat

def fit(q,steps,seed,anchor=1e-5,minsv=0.4,rank_weight=1.):
    torch.set_num_threads(1); torch.manual_seed(seed)
    z=0.43
    f,k=base.target_coefficients(t=1.,q=q)
    f=torch.tensor(f,dtype=DT);k=torch.tensor(k,dtype=DT)
    Pbase,Rbase=base.known_reference_lax(z,tracefree=True)
    Pbase=torch.tensor(Pbase,dtype=DT);Rbase=torch.tensor(Rbase,dtype=DT)
    P0=torch.nn.Parameter(Pbase+0.01*torch.randn_like(Pbase))
    R0=torch.nn.Parameter(Rbase+0.01*torch.randn_like(Rbase))
    P1=torch.nn.Parameter(0.01*torch.randn_like(Pbase))
    R1=torch.nn.Parameter(0.01*torch.randn_like(Rbase))
    xs=[-1.,-.5,0.,.5,1.]
    opt=torch.optim.Adam([P0,R0,P1,R1],lr=.02)
    def criterion():
        rems=[];qranks=[]
        for x in xs:
            C,Q=coefficient_remainder(P0,R0,P1,R1,x,f,k)
            rems.append(C.square().mean())
            qranks.append(torch.relu(minsv-torch.linalg.svdvals(Q)).square().mean())
        err=torch.stack(rems).mean()
        reg=(P0-Pbase).square().mean()+(R0-Rbase).square().mean()+P1.square().mean()+R1.square().mean()
        return err+rank_weight*torch.stack(qranks).mean()+anchor*reg,err
    for i in range(steps):
        opt.zero_grad()
        loss,err=criterion()
        loss.backward();opt.step()
    opt2=torch.optim.LBFGS([P0,R0,P1,R1],lr=.6,max_iter=90,line_search_fn='strong_wolfe')
    def closure():
        opt2.zero_grad();l,_=criterion();l.backward();return l
    opt2.step(closure)
    tensors=tuple(t.detach().clone() for t in (P0,R0,P1,R1))
    def evalpoints(xs):
        out=[]
        for x in xs:
            C,Q=coefficient_remainder(*tensors,float(x),f,k)
            vals=torch.linalg.svdvals(Q)
            out.append(dict(x=x,rms=float(C.square().mean().sqrt()),
                            min_Q_singular=float(vals[-1])))
        return out
    return tensors,evalpoints(xs),evalpoints([-.88,-.23,.17,.79,1.12])

def main(args):
    rows=[]
    for q in [0.,1.]:
        for seed in range(1,args.seeds+1):
            params,train,hold=fit(q,args.steps,seed,anchor=args.anchor_weight,
                                  minsv=args.min_singular,rank_weight=args.rank_weight)
            row=dict(flux=q,seed=seed,train=train,holdout=hold,
                     train_rms=float(np.mean([x['rms'] for x in train])),
                     holdout_rms=float(np.mean([x['rms'] for x in hold])),
                     holdout_min_Q_singular=min(x['min_Q_singular'] for x in hold))
            rows.append(row)
            print('q',q,'seed',seed,'train',row['train_rms'],
                  'holdout',row['holdout_rms'],
                  'min sv',row['holdout_min_Q_singular'],flush=True)
            if args.output:
                Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                np.savez(Path(args.output).with_name('nilpotent_sigma_fqe_xlinear_weights_q'+str(int(q))+'_seed'+str(seed)+'.npz'),
                         **dict(zip(['P0','R0','P1','R1'],[x.numpy() for x in params])))
    if args.output:
       with open(args.output,'w') as fi:
           json.dump(dict(description='off-shell F=QE x-linear coefficient optimization',
                     target='G=I, [e1,e2]=e4,[e1,e3]=e5,H=q(e124+e135)',
                     aux='gl5, traceless reference; single z=0.43',cases=rows),fi,indent=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--seeds',type=int,default=1)
    ap.add_argument('--anchor-weight',type=float,default=1e-5)
    ap.add_argument('--min-singular',type=float,default=.4)
    ap.add_argument('--rank-weight',type=float,default=1.)
    ap.add_argument('--output',default='results/03_nilpotent_fqe/coordinate_linear_results.json')
    main(ap.parse_args())

