import os, numpy, scipy, tempfile


import pyscf
from pyscf import lib, scf, pbc
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

# from pyscf.grad.rks import _scale_ao, _dot_ao_dm, _d1_dot_
# from pyscf.hessian.rks import _make_dR_rho1

import eph
import eph.mol, eph.mol.eph_fd
import eph.mol.rhf, eph.mol.rks

from eph.pbc import eph_fd
from eph.pbc.rhf import ElectronPhononCouplingBase

# def _get_vxc_deriv1(eph_obj, mo_coeff, mo_occ, max_memory=None):
#     cell = eph_obj.cell
#     mf = eph_obj.base
#     if eph_obj.grids is not None:
#         grids = eph_obj.grids
#     else:
#         grids = mf.grids

#     nao, nmo = mo_coeff.shape
#     ni = mf._numint
#     xctype = ni._xc_type(mf.xc)
#     aoslices = cell.aoslice_by_atom()
#     shls_slice = (0, cell.nbas)
#     ao_loc = cell.ao_loc_nr()
#     dm0 = mf.make_rdm1(mo_coeff, mo_occ)

#     vmat = numpy.zeros((cell.natm, 3, nao, nao))
#     max_memory = max(2000, max_memory-vmat.size*8/1e6)

#     if xctype == 'LDA':
#         ao_deriv = 1
#         block_loop = ni.block_loop(cell, grids, nao, ao_deriv, max_memory)
#         for ao, mask, weight, coords in block_loop:
#             rho = ni.eval_rho2(cell, ao[0], mo_coeff, mo_occ, mask, xctype)
#             vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
#             wv = weight * vxc[0]

#             aow = _scale_ao(ao[0], wv)
#             ao_dm0 = _dot_ao_dm(cell, ao[0], dm0, mask, shls_slice, ao_loc)

#             wf = weight * fxc[0,0]
#             for ia in range(cell.natm):
#                 p0, p1 = aoslices[ia][2:]
#                 rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
#                 wv = wf * rho1
#                 aow = [_scale_ao(ao[0], wv[i]) for i in range(3)]
#                 _d1_dot_(vmat[ia], cell, aow, ao[0], mask, ao_loc, True)
#             ao_dm0 = aow = None

#     elif xctype == 'GGA':
#         ao_deriv = 2
#         for ao, mask, weight, coords \
#                 in ni.block_loop(cell, grids, nao, ao_deriv, max_memory):
#             rho = ni.eval_rho2(cell, ao[:4], mo_coeff, mo_occ, mask, xctype)
#             vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
#             wv = weight * vxc
#             wv[0] *= .5

#             ao_dm0 = [_dot_ao_dm(cell, ao[i], dm0, mask, shls_slice, ao_loc)
#                       for i in range(4)]
#             wf = weight * fxc
#             for ia in range(cell.natm):
#                 dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
#                 wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
#                 wv[:,0] *= .5
#                 aow = [_scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
#                 _d1_dot_(vmat[ia], cell, aow, ao[0], mask, ao_loc, True)
#             ao_dm0 = aow = None

#     elif xctype == 'MGGA':
#         raise NotImplementedError
    
#     vmat = numpy.stack([vmat, vmat], axis=0)
#     for ia, (s0, s1, p0, p1) in enumerate(aoslices):
#         vmat[0, ia, :, p0:p1] += v_ip[:, p0:p1]
#     vmat = - vmat - vmat.transpose(0, 1, 2, 4, 3)
#     return vmat
    
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, pyscf.pbc.dft.rks.RKS)
        assert not isinstance(method, pyscf.pbc.scf.khf.KSCF)
        ElectronPhononCouplingBase.__init__(self, method)
        self.grids = method.grids

    def solve_mo1(self, vresp, s1=None, f1=None, mo_energy=None,
                    mo_coeff=None, mo_occ=None, verbose=None):
        
        log = logger.new_logger(self, verbose)
        from eph.mol.rhf import solve_mo1
        return solve_mo1(
            vresp, s1=s1, f1=f1, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ, verbose=log,
            max_cycle=self.max_cycle, tol=self.conv_tol_cphf,
            level_shift=self.level_shift
        )
    
    def gen_vxc_deriv(self, mo_coeff, mo_occ):
        scf_obj = self.base
        cell_obj = scf_obj.cell

        ni = scf_obj._numint
        ni.libxc.test_deriv_order(
            scf_obj.xc, 2, raise_error=True
        )

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            scf_obj.xc, spin=cell_obj.spin
        )

        is_hybrid = ni.libxc.is_hybrid_xc(scf_obj.xc)

        from eph.mol.rks import _get_vxc_deriv1
        vxc1 = _get_vxc_deriv1(
            self, mo_coeff, mo_occ,
            max_memory=self.max_memory
        )

        return vxc1, (omega, alpha, hyb, is_hybrid)
    
    def gen_fock_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        scf_obj = self.base

        mol_obj = scf_obj.mol
        aoslices = mol_obj.aoslice_by_atom()
        df_obj = scf_obj.with_df

        nbas = mol_obj.nbas
        nao, nmo = mo_coeff.shape
        orbo = mo_coeff[:, mo_occ > 0]
        nocc = orbo.shape[1]
        nvir = nmo - nocc

        dm0 = scf_obj.make_rdm1(mo_coeff, mo_occ)

        tmp = self.gen_vxc_deriv(mo_coeff, mo_occ)
        vxc_deriv = tmp[0]
        omega, alpha, hyb, is_hybrid = tmp[1]
        assert not is_hybrid

        hcor_deriv = scf_obj.nuc_grad_method().hcore_generator()

        def func(ia):
            h1 = hcor_deriv(ia)

            s0, s1, p0, p1 = aoslices[ia] 
            from jk import _get_j1_lk_ij
            from jk import _get_j1_ji_kl

            vj1 = _get_j1_ji_kl(df_obj, (p0, p1), -dm0[:, p0:p1])
            vj2 = _get_j1_lk_ij(df_obj, (p0, p1), -dm0)

            jk1 = vj1 + vj1.transpose(0, 2, 1)
            vjk = vj1
            vjk[:, p0:p1] += vj2
            vjk += vjk.transpose(0, 2, 1)

            f1 = h1 + vjk
            if vxc_deriv is not None:
                f1  += vxc_deriv[0, ia]
                jk1 += vxc_deriv[1, ia]
            return f1, jk1

        return func

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft import multigrid

    from ase.build import bulk
    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

    a = 3.5668
    diamond = bulk("C", "diamond", a=a)

    cell = gto.Cell()
    cell.atom = ase_atoms_to_pyscf(diamond)
    cell.a = diamond.cell
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.unit = 'A'
    cell.verbose = 4
    cell.ke_cutoff = 100
    cell.build()

    mf = scf.RKS(cell)
    mf.xc = "PBE"
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel(dm0=None)
    dm0 = mf.make_rdm1()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol = eph_obj.kernel()
