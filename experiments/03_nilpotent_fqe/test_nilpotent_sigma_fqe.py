"""Reproducible identity and consistency tests for the N5 off-shell F=Q(E) scripts."""
import numpy as np
import torch
import nilpotent_sigma_fqe_optimization as m
import nilpotent_sigma_fqe_coordinate_dependent as field


def test_reference_lax_is_offshell_exact_for_multiple_spectral_parameters():
    for tracefree in (False,True):
        for z in (-0.72,-0.43,0.23,0.43,0.61):
            P,R=m.known_reference_lax(z,tracefree=tracefree)
            f,k=m.target_coefficients(t=0.,q=0.)
            c=m.remainder_numpy(P,R,f,k)
            assert np.max(np.abs(c))<1e-12
            assert m.offshell_test(P,R,f,k,samples=32)<1e-11
            assert np.linalg.matrix_rank(m.qmap_numpy(P,R),tol=1e-8)==5


def test_coordinate_dependent_offshell_jet_identity():
    rng=np.random.default_rng(20261101)
    P0,R0=m.known_reference_lax(.43,tracefree=True)
    P0=P0+0.02*rng.normal(size=P0.shape)
    R0=R0+0.02*rng.normal(size=R0.shape)
    P1=0.02*rng.normal(size=P0.shape)
    R1=0.02*rng.normal(size=R0.shape)
    f,k=m.target_coefficients(t=1.,q=1.)
    rng=np.random.default_rng(20261101)
    for x in [-0.89,-0.21,0.12,0.74]:
        Pt=P0+x*P1;Rt=R0+x*R1
        Ct,Q=field.coefficient_remainder(*[torch.tensor(a,dtype=torch.float64) for a in (P0,R0,P1,R1)],x,
                                         torch.tensor(f,dtype=torch.float64),torch.tensor(k,dtype=torch.float64))
        C=Ct.numpy()
        for _ in range(8):
            up=rng.normal(size=5);um=rng.normal(size=5); E=rng.normal(size=5)
            fb=np.einsum('cab,a,b->c',f,up,um)
            kb=np.einsum('cab,a,b->c',k,up,um)
            dpu=(E-kb-fb)/2; dmu=(E-kb+fb)/2
            comm=(Pt[:,None]@Rt[None,:]-Rt[None,:]@Pt[:,None])
            curvature=np.einsum('a,aij->ij',dpu,Rt)-np.einsum('a,aij->ij',dmu,Pt)
            curvature+=np.einsum('a,b,abij->ij',up,um,comm)
            curvature+=up[0]*np.einsum('b,bij->ij',um,R1)-um[0]*np.einsum('a,aij->ij',up,P1)
            QonE=np.einsum('a,aij->ij',E,(Rt-Pt)/2)
            predicted=np.einsum('a,b,abij->ij',up,um,C)
            np.testing.assert_allclose(curvature-QonE,predicted,rtol=1e-10,atol=1e-10)


def test_qloss_not_identical_to_onshell_flatness():
    f,k=m.target_coefficients(1.,1.)
    P,R=m.known_reference_lax(.43,tracefree=True)
    assert m.offshell_test(P,R,f,k)>1.

