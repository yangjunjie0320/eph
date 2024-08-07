import os, numpy, scipy, tempfile

import eph.mol
import pyscf
from pyscf import lib, scf, dft
import pyscf.eph
import pyscf.hessian

from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad
from pyscf.hessian.rks import _make_dR_rho1

import eph
from eph.mol import eph_fd, rhf
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

def _get_vxc_deriv(mo_occ=None, mo_coeff=None, scf_obj=None, max_memory=2000, verbose=None):
    log = logger.new_logger(None, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    # information from mol object
    mol = scf_obj.mol
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # grids information
    grids = scf_obj.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    # information from scf object
    nao, nmo = mo_coeff.shape
    ni = scf_obj._numint
    xc = scf_obj.xc
    xctype = ni._xc_type(xc)
    dm0 = scf_obj.make_rdm1(mo_coeff, mo_occ)

    # allocate memory for vmat
    vmat = numpy.zeros((natm, 3, nao, nao))
    max_memory = max(2000, max_memory - vmat.size * 8 / 1e6)

    if xctype == 'LDA':
        ao_deriv = 1
        block_loop = ni.block_loop(mol, grids, nao, ao_deriv, max_memory)
        for ao, mask, weight, coords in block_loop:
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype)[1:3]

            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0, 0]

            for ia, (s0, s1, p0, p1) in enumerate(aoslices):
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:, :, p0:p1], ao_dm0[:, p0:p1])
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        block_loop = ni.block_loop(mol, grids, nao, ao_deriv, max_memory)
        for ao, mask, weight, coords in block_loop:
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype)[1:3]

            wv = weight * vxc
            wv[0] *= 0.5

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc

            for ia, (s0, s1, p0, p1) in enumerate(aoslices):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:, 0] *= 0.5
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    t1 = log.timer_debug1('vxc', *t0)
    return -(vmat + vmat.transpose(0, 1, 3, 2))

def gen_veff_deriv(mo_occ=None, mo_coeff=None, scf_obj=None, mo1=None, h1ao=None, verbose=None):
    log = logger.new_logger(None, verbose)

    mol = scf_obj.mol
    aoslices = mol.aoslice_by_atom()
    nao, nmo = mo_coeff.shape
    nbas = scf_obj.mol.nbas

    orbo = mo_coeff[:, mo_occ > 0]
    nocc = orbo.shape[1]
    dm0 = numpy.dot(orbo, orbo.T) * 2.0
    assert isinstance(scf_obj, scf.hf.RHF) or isinstance(scf_obj, scf.rks.RKS)

    if isinstance(scf_obj, dft.rks.KohnShamDFT):
        # test if the functional has the second derivative
        ni = scf_obj._numint
        ni.libxc.test_deriv_order(
            scf_obj.xc, 2, raise_error=True
        )

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            scf_obj.xc, spin=mol.spin
        )

        is_hybrid = ni.libxc.is_hybrid_xc(scf_obj.xc)

        vxc1 = _get_vxc_deriv(
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            scf_obj=scf_obj,
            max_memory=2000, verbose=verbose
            )
        
    else: # is Hartree-Fock
        assert isinstance(scf_obj, scf.hf.RHF)
        omega, alpha, hyb = 0.0, 0.0, 1.0
        is_hybrid = True
        vxc1 = None

    vresp = scf_obj.gen_response(mo_coeff, mo_occ, hermi=1)

    def load(ia):
        assert h1ao is not None
        assert mo1 is not None

        t1 = None
        if isinstance(mo1, str):
            assert os.path.exists(mo1), '%s not found' % mo1
            t1 = lib.chkfile.load(mo1, 'scf_mo1/%d' % ia)
            t1 = t1.reshape(-1, nao, nocc)

        else:
            t1 = mo1[ia].reshape(-1, nao, nocc)

        assert t1 is not None

        from pyscf.hessian.rhf import _get_jk
        s0, s1, p0, p1 = aoslices[ia]
        shls_slice  = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s2kl', -dm0[:, p0:p1]] # vj1
        script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1
        
        if is_hybrid:
            
            tmp = _get_jk(
                mol, 'int2e_ip1', 3, 's2kl',
                script_dms=script_dms,
                shls_slice=shls_slice
            )

            vj1, vk1 = tmp
            vjk1 = vj1 - vk1 * 0.5 * hyb

            if omega != 0.0:
                with mol.with_range_coulomb(omega):
                    vk1 = _get_jk(
                        mol, 'int2e_ip1', 3, 's2kl',
                        script_dms=script_dms[2:],
                        shls_slice=shls_slice
                    )
                vjk1 -= (alpha - hyb) * 0.5 * vk1

        else: # is pure functional
            vj1 = _get_jk(
                mol, 'int2e_ip1', 3, 's2kl',
                script_dms=script_dms[:2],
                shls_slice=shls_slice
            )

            vjk1 = vj1

        return t1, vjk1

    def func(ia):
        t1, vjk1 = load(ia)
        dm1 = 2.0 * numpy.einsum('xpi,qi->xpq', t1, orbo)
        dm1 = dm1 + dm1.transpose(0, 2, 1)

        v1  = vjk1 + vjk1.transpose(0, 2, 1)
        v1 += vresp(dm1)

        if vxc1 is not None:
            v1 += vxc1[ia]
        
        return v1

    return func

class ElectronPhononCoupling(eph.mol.rhf.ElectronPhononCoupling):
    def __init__(self, method):
        assert isinstance(method, pyscf.dft.rks.RKS)
        ElectronPhononCouplingBase.__init__(self, method)
        self.grids = method.grids
        self.grid_response = False
    
if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O      -0.0000000000    -0.0000000000     0.1190217807
    H      -0.7590867860     0.0000000000    -0.4760951386
    H       0.7590867860     0.0000000000    -0.4760951386
    '''
    mol.basis = '631g*'
    mol.verbose = 0
    mol.symmetry = False
    mol.cart = True
    mol.unit = "AA"
    mol.build()

    natm = mol.natm
    nao = mol.nao_nr()

    mf = scf.RKS(mol)
    mf.xc = "pbe0"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-3)
    hess_obj = mf.Hessian()
    hess_obj.chkfile = mf.chkfile
    hess = hess_obj.kernel()

    from eph.mol import rks
    eph_obj = rks.ElectronPhononCoupling(mf)
    dv_sol = eph_obj.kernel()
    
    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-3)
    hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))
