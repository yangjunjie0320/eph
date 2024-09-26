import numpy, scipy, sys, os

import pyscf
from pyscf import lib
from pyscf.pbc import gto, tools
from pyscf.pbc.gto import mole

from pyscf.pbc.gto.pseudo.pp import _qli
from pyscf.pbc.dft.numint import eval_ao_kpts
from pyscf.pbc.grad.krhf import _make_fakemol
from pyscf.pbc.gto.pseudo.pp import get_vlocG
from pyscf.pbc.gto.pseudo.pp import get_alphas

def _ip_vloc(cell, v=None, phi=None, kpt=None):
    mesh = cell.mesh
    ng = numpy.prod(mesh)
    nao = cell.nao_nr()

    assert phi.shape == (4, ng, nao)
    assert v.shape == (ng,)

    phi1 = phi[1:].conj()
    phi2 = phi[0]
    ip_vloc = numpy.einsum('xgm,gn,g->xmn', phi1, v, phi2, optimize=True)
    return ip_vloc


def _pylm(l, rl, nl, gk):
    assert nl > 0
    gn = lib.norm(gk, axis=1)
    fakemol = _make_fakemol()
    env = mole.PTR_ENV_START
    ang = mole.ANG_OF
    pi125 = numpy.pi ** 1.25
    
    fakemol._bas[0, ang] = l
    fakemol._env[env + 3] = .5 * rl ** 2
    fakemol._env[env + 4] = rl ** (l + 1.5) * pi125

    qkl = numpy.array([_qli(gn * rl, l, k) for k in range(nl)])
    pylm = (fakemol.eval_gto('GTOval', gk)).T[:, None, :] * qkl[None, :, :]
    pylm = pylm.transpose(0, 2, 1)

    return pylm

def _ip_vnlp(cell, phi=None, v=None, kpt=None):
    natm = cell.natm
    mesh = cell.mesh
    ng = numpy.prod(mesh)
    nao = cell.nao_nr()

    assert phi.shape == (4, ng, nao)

    gk = cell.Gv + kpt
    v = v.reshape(-1, ng)

    ip_vnlp = numpy.zeros((3, nao, nao), dtype=numpy.complex128)
    for ia in range(natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        pp = cell._pseudo[symb]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl <= 0:
                continue
            
            pylm = _pylm(l, rl, nl, gk)
            z = numpy.einsum('g,img,xgp->ximp', v[ia].conj(), pylm, phi, optimize=True)
            ip_vnlp += numpy.einsum("imp,ij,xjmp->xpq", z[0].conj(), hl, z[1:], optimize=True)

    return ip_vnlp / (ng * ng)

def get_ip_hcore(cell, kpts):
    dh = cell.pbc_intor('int1e_ipkin', kpts=kpts)
    dh = numpy.asarray(dh)

    kpts = numpy.asarray(kpts).reshape(-1, 3)
    nk = kpts.shape[0]
    nao = cell.nao_nr()
    assert dh.shape == (nk, 3, nao, nao)

    mesh = cell.mesh
    ng = numpy.prod(mesh)

    if not cell._pseudo:
        raise NotImplementedError
    
    si=cell.get_SI()
    Gv = cell.Gv
    natom = cell.natm
    coords = cell.get_uniform_grids()
    ngrids = len(coords)
    vlocG = get_vlocG(cell)
    vpplocG = -numpy.einsum('ij,ij->j', si, vlocG)
    vlocG[0] = numpy.sum(get_alphas(cell))
    vloc_r = tools.ifft(vlocG, cell.mesh).real
    fakemol = _make_fakemol()
    ptr = mole.PTR_ENV_START

    for kn, kpt in enumerate(kpts):
        theta = numpy.dot(coords, kpt)
        phase = numpy.exp(-1j * theta)

        phi_r = eval_ao_kpts(cell, coords, kpt, deriv=1)[0] # how to cut into pieces?
        phi_r = phi_r.transpose(0, 2, 1).reshape(-1, ng)
        phi_r = numpy.asarray(phi_r, order='C')
        phi_g = tools.fftk(phi_r, mesh, phase)

        phi_r = phi_r.reshape(4, nao, ng).transpose(0, 2, 1)
        phi_g = phi_g.reshape(4, nao, ng).transpose(0, 2, 1)

        print(vloc_r.shape)
        ip_vloc = _ip_vloc(cell, v=vloc_r, phi=phi_r, kpt=kpt)
        ip_vnlp = _ip_vnlp(cell, v=si,     phi=phi_g, kpt=kpt)

        if dh.dtype == numpy.float64:
            ip_vloc = ip_vloc.real
            ip_vnlp = ip_vnlp.real
        dh[kn, :] += ip_vloc + ip_vnlp
    return dh


# def gen_hcore_deriv(cell=None, kpts=None, with_basis_response=False):
#     if cell is None: cell = mf.cell
#     if kpts is None: kpts = mf.kpts
#     h1 = get_hcore(cell, kpts)
#     dtype = h1.dtype

#     aoslices = cell.aoslice_by_atom()
#     SI=cell.get_SI()  ##[natom ,grid]
#     mesh = cell.mesh
#     Gv = cell.Gv    ##[grid, 3]
#     ngrids = len(Gv)
#     coords = cell.get_uniform_grids()
#     vlocG = get_vlocG(cell)  ###[natom, grid]
#     ptr = mole.PTR_ENV_START
#     def hcore_deriv(atm_id):
#         shl0, shl1, p0, p1 = aoslices[atm_id]
#         symb = cell.atom_symbol(atm_id)
#         fakemol = _make_fakemol()
#         vloc_g = 1j * numpy.einsum('ga,g->ag', Gv, SI[atm_id]*vlocG[atm_id])
#         nkpts, nao = h1.shape[0], h1.shape[2]
#         hcore = numpy.zeros([3,nkpts,nao,nao], dtype=h1.dtype)
#         for kn, kpt in enumerate(kpts):

#             ao = eval_ao_kpts(cell, coords, kpt)[0]
#             rho = numpy.einsum('gi,gj->gij',ao.conj(),ao)
#             for ax in range(3):
#                 vloc_R = tools.ifft(vloc_g[ax], mesh).real
#                 vloc = numpy.einsum('gij,g->ij', rho, vloc_R)
#                 hcore[ax,kn] += vloc
#             rho = None
#             aokG= tools.fftk(numpy.asarray(ao.T, order='C'),
#                               mesh, numpy.exp(-1j*numpy.dot(coords, kpt))).T
#             ao = None
#             Gk = Gv + kpt
#             G_rad = lib.norm(Gk, axis=1)
#             if symb not in cell._pseudo: continue
#             pp = cell._pseudo[symb]
#             for l, proj in enumerate(pp[5:]):
#                 rl, nl, hl = proj
#                 if nl >0:
#                     hl = numpy.asarray(hl)
#                     fakemol._bas[0,mole.ANG_OF] = l
#                     fakemol._env[ptr+3] = .5*rl**2
#                     fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
#                     pYlm_part = fakemol.eval_gto('GTOval', Gk)
#                     pYlm = numpy.empty((nl,l*2+1,ngrids))
#                     for k in range(nl):
#                         qkl = _qli(G_rad*rl, l, k)
#                         pYlm[k] = pYlm_part.T * qkl
#                     SPG_lmi = numpy.einsum('g,nmg->nmg', SI[atm_id].conj(), pYlm)
#                     SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
#                     SPG_lmi_G = 1j * numpy.einsum('nmg, ga->anmg', SPG_lmi, Gv)
#                     SPG_lm_G_aoG = numpy.einsum('anmg, gp->anmp', SPG_lmi_G, aokG)
#                     tmp_1 = numpy.einsum('ij,ajmp->aimp', hl, SPG_lm_G_aoG)
#                     tmp_2 = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
#                     vppnl = (numpy.einsum('imp,aimq->apq', SPG_lm_aoG.conj(), tmp_1) +
#                              numpy.einsum('aimp,imq->apq', SPG_lm_G_aoG.conj(), tmp_2))
#                     vppnl *=(1./ngrids**2)
#                     if dtype==numpy.float64:
#                         hcore[:,kn] += vppnl.real
#                     else:
#                         hcore[:,kn] += vppnl
#             hcore[:,kn,p0:p1] -= h1[kn,:,p0:p1]
#             hcore[:,kn,:,p0:p1] -= h1[kn,:,p0:p1].transpose(0,2,1).conj()
#         return hcore
#     return hcore_deriv

if __name__ == "__main__":
    from ase.build import bulk
    diamond = bulk('C', 'diamond', a=3.567)

    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
    cell = gto.Cell()
    cell.atom = ase_atoms_to_pyscf(diamond)
    cell.a = diamond.cell
    cell.unit = 'A'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [11] * 3
    cell.build()

    ip_hcore = get_ip_hcore(cell, kpts=numpy.zeros((1, 3)))
    print(ip_hcore.shape)
    
    