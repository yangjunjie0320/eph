import os, numpy, scipy, tempfile

import eph.mol
import eph.mol.eph_fd
import eph.mol.rhf
import pyscf
from pyscf import lib, scf
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
import pyscf.pbc
from pyscf.scf import hf, _vhf
from pyscf import hessian

import eph
from eph.pbc import eph_fd
from eph.pbc.rhf import ElectronPhononCouplingBase

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    v_ip = numpy.zeros((3,nao,nao))
    vmat = numpy.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'MGGA':
        raise NotImplementedError
    
    vmat = numpy.stack([vmat, vmat], axis=0)
    for ia, (s0, s1, p0, p1) in enumerate(aoslices):
        vmat[0, ia, :, p0:p1] += v_ip[:, p0:p1]
    vmat = - vmat - vmat.transpose(0, 1, 2, 4, 3)
    return vmat
    
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

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
        return None, (0.0, 0.0, 1.0, True)
    
    def gen_fock_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        scf_obj = self.base

        mol_obj = scf_obj.mol
        aoslices = mol_obj.aoslice_by_atom()

        nbas = mol_obj.nbas
        nao, nmo = mo_coeff.shape
        orbo = mo_coeff[:, mo_occ > 0]
        nocc = orbo.shape[1]
        nvir = nmo - nocc

        dm0 = scf_obj.make_rdm1(mo_coeff, mo_occ)

        tmp = self.gen_vxc_deriv(mo_coeff, mo_occ)
        vxc_deriv = tmp[0]
        omega, alpha, hyb, is_hybrid = tmp[1]
        # assert not is_hybrid

        grad_obj = scf_obj.nuc_grad_method()
        hcor_deriv = scf_obj.nuc_grad_method().hcore_generator()

        def func(ia):
            h1 = hcor_deriv(ia)

            s0, s1, p0, p1 = aoslices[ia]
            shls_slice = (s0, s1) + (0, nbas) * 3

            script_dms  = ['ji->s1kl', -dm0[:, p0:p1]] # vj1
            script_dms += ['lk->s1ij', -dm0]           # vj2

            from jk import _get_jk #  shall be replaced with faster functions
            if not is_hybrid:
                vj1, vj2 = _get_jk(
                    mol_obj, 'int2e_ip1', 3, 's1',
                    script_dms=script_dms,
                    shls_slice=shls_slice
                )

                jk1 = vj1 + vj1.transpose(0, 2, 1)
                vjk = vj1
                vjk[:, p0:p1] += vj2
                vjk += vjk.transpose(0, 2, 1)

            else:
                script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1
                script_dms += ['jk->s1il', -dm0]           # vk2
            
                vj1, vj2, vk1, vk2 = _get_jk(
                    mol_obj, 'int2e_ip1', 3, 's1',
                    script_dms=script_dms,
                    shls_slice=shls_slice
                )

                jk1 = vj1 - hyb * 0.5 * vk1
                jk1 = jk1 + jk1.transpose(0, 2, 1)

                vjk = vj1 - hyb * 0.5 * vk1
                vjk[:, p0:p1] += vj2 - hyb * 0.5 * vk2
                vjk += vjk.transpose(0, 2, 1)

            f1 = h1 + vjk
            if vxc_deriv is not None:
                f1 += vxc_deriv[0, ia]
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




    # eph = ElectronPhononCouplingBase(mf)
    # vnuc_deriv = eph.gen_vnuc_deriv()
    # ovlp_deriv = eph.gen_ovlp_deriv()