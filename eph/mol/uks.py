import os, numpy, scipy, tempfile

import eph.mol
import pyscf
from pyscf import lib, scf, dft
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from pyscf.grad import rks as rks_grad
from pyscf.hessian import rks as rks_hess
from pyscf.dft import numint, gen_grid
from pyscf.hessian.rks import _make_dR_rho1, _check_mgga_grids

import eph
from eph.mol import eph_fd, uhf
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

    nao, nmo = mo_coeff[0].shape
    ni = scf_obj._numint
    xc = scf_obj.xc
    xctype = ni._xc_type(xc)
    dm0a, dm0b = scf_obj.make_rdm1(mo_coeff, mo_occ)

    vmata = numpy.zeros((natm, 3, nao, nao))
    vmatb = numpy.zeros((natm, 3, nao, nao))

    max_memory -= vmata.size * 8/1e6 + vmatb.size * 8/1e6
    max_memory = max(2000, max_memory)

    if xctype == 'LDA':
        ao_deriv = 1
        block_loop = ni.block_loop(mol, grids, nao, ao_deriv, max_memory)
        for ao, mask, weight, coords in block_loop:
            rhoa = ni.eval_rho2(mol, ao[0], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[0], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(scf_obj.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]

            wv = weight * vxc[:,0]
            ao_dm0a = numint._dot_ao_dm(mol, ao[0], dm0a, mask, shls_slice, ao_loc)
            ao_dm0b = numint._dot_ao_dm(mol, ao[0], dm0b, mask, shls_slice, ao_loc)
            aow1a = numpy.einsum('xpi,p->xpi', ao[1:], wv[0])
            aow1b = numpy.einsum('xpi,p->xpi', ao[1:], wv[1])
            wf = weight * fxc[:,0,:,0]

            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                rho1a = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0a[:,p0:p1])
                rho1b = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0b[:,p0:p1])
                wv  = wf[0,:,None] * rho1a
                wv += wf[1,:,None] * rho1b
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[0])
                aow[:,:,p0:p1] += aow1a[:,:,p0:p1]
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = numpy.einsum('pi,xp->xpi', ao[0], wv[1])
                aow[:,:,p0:p1] += aow1b[:,:,p0:p1]
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = aow1a = aow1b = None

        for ia in range(mol.natm):
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        vipa = numpy.zeros((3,nao,nao))
        vipb = numpy.zeros((3,nao,nao))
        block_loop = ni.block_loop(mol, grids, nao, ao_deriv, max_memory)
        for ao, mask, weight, coords in block_loop:
            rhoa = ni.eval_rho2(mol, ao[:4], mo_coeff[0], mo_occ[0], mask, xctype)
            rhob = ni.eval_rho2(mol, ao[:4], mo_coeff[1], mo_occ[1], mask, xctype)
            vxc, fxc = ni.eval_xc_eff(scf_obj.xc, (rhoa, rhob), 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[:,0] *= .5
            rks_grad._gga_grad_sum_(vipa, mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vipb, mol, ao, wv[1], mask, ao_loc)

            ao_dm0a = [numint._dot_ao_dm(mol, ao[i], dm0a, mask, shls_slice, ao_loc) for i in range(4)]
            ao_dm0b = [numint._dot_ao_dm(mol, ao[i], dm0b, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1a = rks_hess._make_dR_rho1(ao, ao_dm0a, ia, aoslices, xctype)
                dR_rho1b = rks_hess._make_dR_rho1(ao, ao_dm0b, ia, aoslices, xctype)
                wv  = numpy.einsum('xbyg,sxg->bsyg', wf[0], dR_rho1a)
                wv += numpy.einsum('xbyg,sxg->bsyg', wf[1], dR_rho1b)
                wv[:,:,0] *= .5
                wva, wvb = wv

                aow = [numint._scale_ao(ao[:4], wva[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmata[ia], mol, aow, ao[0], mask, ao_loc, True)
                aow = [numint._scale_ao(ao[:4], wvb[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmatb[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0a = ao_dm0b = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmata[ia,:,p0:p1] += vipa[:,p0:p1]
            vmatb[ia,:,p0:p1] += vipb[:,p0:p1]
            vmata[ia] = -vmata[ia] - vmata[ia].transpose(0,2,1)
            vmatb[ia] = -vmatb[ia] - vmatb[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmata, vmatb

def gen_veff_deriv(mo_occ=None, mo_coeff=None, scf_obj=None, mo1=None, h1ao=None, verbose=None):
    log = logger.new_logger(None, verbose)

    mol = scf_obj.mol
    aoslices = mol.aoslice_by_atom()
    nao, nmo = mo_coeff[0].shape
    nbas = mol.nbas
    
    orboa = mo_coeff[0][:, mo_occ[0] > 0]
    orbob = mo_coeff[1][:, mo_occ[1] > 0]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]
    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    assert isinstance(scf_obj, scf.uhf.UHF) or isinstance(scf_obj, scf.uks.UKS)

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
        assert isinstance(scf_obj, scf.uhf.UHF)
        omega, alpha, hyb = 0.0, 0.0, 1.0
        is_hybrid = True
        vxc1 = None

    vresp = scf_obj.gen_response(mo_coeff, mo_occ, hermi=1)
    
    def load(ia):
        assert h1ao is not None
        assert mo1 is not None

        t1a = None
        t1b = None
        if isinstance(mo1, str):
            assert os.path.exists(mo1), '%s not found' % mo1
            t1a = lib.chkfile.load(mo1, 'scf_mo1/0/%d' % ia)
            t1b = lib.chkfile.load(mo1, 'scf_mo1/1/%d' % ia)
            t1a = t1a.reshape(-1, nao, nocca)
            t1b = t1b.reshape(-1, nao, noccb)

        else:
            mo1a, mo1b = mo1
            t1a = mo1a[ia].reshape(-1, nao, nocca)
            t1b = mo1b[ia].reshape(-1, nao, noccb)

        assert t1a is not None
        assert t1b is not None
        t1 = (t1a, t1b)

        from pyscf.hessian.rhf import _get_jk
        s0, s1, p0, p1 = aoslices[ia]
        shls_slice  = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s2kl', -dm0a[:,p0:p1]] # vj1a
        script_dms += ['ji->s2kl', -dm0b[:,p0:p1]] # vj1b
        script_dms += ['li->s1kj', -dm0a[:,p0:p1]] # vk1a
        script_dms += ['li->s1kj', -dm0b[:,p0:p1]] # vk1b

        if is_hybrid:
            tmp = _get_jk(
                mol, 'int2e_ip1', 3, 's2kl',
                script_dms=script_dms,
                shls_slice=shls_slice
            )
            
            vj1a, vj1b, vk1a, vk1b = tmp
            vjk1a = vj1a + vj1b - vk1a * hyb
            vjk1b = vj1a + vj1b - vk1b * hyb

            if omega != 0.0:
                with mol.with_range_couloomb(omega):
                    vk1a, vk1b = _get_jk(
                        mol, 'int2e_ip1', 3, 's2kl',
                        script_dms=script_dms[2:],
                        shls_slice=shls_slice
                    )

                vjk1a -= (alpha - hyb) * vk1a
                vjk1b -= (alpha - hyb) * vk1b

        else: # is pure functional
            vj1a, vj1b = _get_jk(
                mol, 'int2e_ip1', 3, 's2kl',
                script_dms=script_dms[:4],
                shls_slice=shls_slice
            )

            vj1 = vj1a + vj1b
            vjk1a = vjk1b = vj1

        return t1, (vjk1a, vjk1b)
    
    def func(ia):
        (t1a, t1b), (vjk1a, vjk1b) = load(ia)
        dm1a = numpy.einsum('xpi,qi->xpq', t1a, orboa)
        dm1a += dm1a.transpose(0, 2, 1)

        dm1b = numpy.einsum('xpi,qi->xpq', t1b, orbob)
        dm1b += dm1b.transpose(0, 2, 1)
        dm1 = numpy.asarray((dm1a, dm1b))

        v1a, v1b = vresp(dm1)
        v1a += vjk1a + vjk1a.transpose(0, 2, 1)
        v1b += vjk1b + vjk1b.transpose(0, 2, 1)

        if vxc1 is not None:
            v1a += vxc1[0][ia]
            v1b += vxc1[1][ia]
        
        return (v1a, v1b)
    
    return func

class ElectronPhononCoupling(eph.mol.uhf.ElectronPhononCoupling):
    def __init__(self, method):
        assert isinstance(method, pyscf.dft.uks.UKS)
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

    mf = scf.UKS(mol)
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

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))
