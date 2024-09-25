import os, numpy, scipy, tempfile

import eph.mol
import pyscf
from pyscf import lib, scf, dft

from pyscf.grad import rks as rks_grad
from pyscf.dft import numint
from pyscf.hessian import rks as rks_hess
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian.rks import _make_dR_rho1

import eph
from eph.mol import eph_fd, rhf
from eph.mol.rhf import ElectronPhononCouplingBase

def _get_vxc_deriv1(eph_obj, mo_coeff, mo_occ, max_memory):
    mol = eph_obj.mol
    mf = eph_obj.base
    if eph_obj.grids is not None:
        grids = eph_obj.grids
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
        block_loop = ni.block_loop(mol, grids, nao=nao, deriv=ao_deriv, max_memory=max_memory)
        for tmp in block_loop:
            ao = tmp[0]
            mask = tmp[-3]
            weight = tmp[-2]

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
        block_loop = ni.block_loop(mol, grids, nao=nao, deriv=ao_deriv, max_memory=max_memory)
        for tmp in block_loop:
            ao = tmp[0]
            mask = tmp[-3]
            weight = tmp[-2]
            
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

class ElectronPhononCoupling(eph.mol.rhf.ElectronPhononCoupling):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)
        self.grids = method.grids

    def gen_vxc_deriv(self, mo_coeff, mo_occ):
        scf_obj = self.base
        mol_obj = scf_obj.mol

        ni = scf_obj._numint
        ni.libxc.test_deriv_order(
            scf_obj.xc, 2, raise_error=True
        )

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            scf_obj.xc, spin=mol_obj.spin
        )

        is_hybrid = ni.libxc.is_hybrid_xc(scf_obj.xc)

        vxc1 = _get_vxc_deriv1(
            self, mo_coeff, mo_occ, 
            max_memory=self.max_memory
        )

        return vxc1, (omega, alpha, hyb, is_hybrid)
    
if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       0.0000000000     0.0000000000     0.1146878262
    H      -0.7540663886    -0.0000000000    -0.4587203947
    H       0.7540663886    -0.0000000000    -0.4587203947
    '''
    mol.basis = 'sto3g' # 631g*'
    mol.verbose = 0
    mol.symmetry = False
    mol.cart = True
    mol.unit = "AA"
    mol.build()

    natm = mol.natm
    nao = mol.nao_nr()

    mf = scf.RKS(mol)
    mf.xc = "PBE0"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-3)
    # hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.grids = None
    dv_sol  = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))