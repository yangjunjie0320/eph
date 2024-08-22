import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

import eph
import eph.mol.eph_fd
from eph.mol.eph_fd import harmonic_analysis

def kernel(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None,
           chkfile=None, atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol
    scf_obj = eph_obj.base

    if mo_energy is None: mo_energy = scf_obj.mo_energy
    if mo_occ    is None: mo_occ    = scf_obj.mo_occ
    if mo_coeff  is None: mo_coeff  = scf_obj.mo_coeff
    if atmlst is None:    atmlst    = range(mol_obj.natm)
    nao, nmo = mo_coeff.shape[-2:]

    chkfile = chkfile if chkfile is not None else eph_obj.chkfile
    eph_obj.solve_mo1(
        mo_energy=mo_energy, mo_coeff=mo_coeff, 
        mo_occ=mo_occ, chkfile=chkfile,
        atmlst=atmlst, verbose=log
        )
    assert os.path.exists(chkfile), '%s not found' % chkfile

    from pyscf import dft
    if isinstance(scf_obj, dft.rks.KohnShamDFT):
        # test if the functional has the second derivative
        ni = scf_obj._numint
        ni.libxc.test_deriv_order(
            scf_obj.xc, 2, raise_error=True
        )

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            scf_obj.xc, spin=mol_obj.spin
        )

        is_hybrid = ni.libxc.is_hybrid_xc(scf_obj.xc)
        
    else: # is Hartree-Fock
        assert isinstance(scf_obj, scf.hf.RHF)
        omega, alpha, hyb = 0.0, 0.0, 1.0
        is_hybrid = True
        vxc1 = None

    assert omega == 0.0

    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol_obj)
    vresp = scf_obj.gen_response(mo_coeff, mo_occ, hermi=1)
    orbo = mo_coeff[:, mo_occ > 0]
    nocc = orbo.shape[1]
    
    dv_ao = [] # numpy.zeros((len(atmlst), 3, nao, nao))
    for i0, ia in enumerate(atmlst):
        # what I need?
        h1 = None
        f1 = None
        s1 = None

        dm1 = None
        # what is vjk1?
        # f1 s1 are only used for cphf, not for the vjk1

        dm1 = lib.chkfile.load(chkfile, 'scf_dm1/%d' % ia)

        vjk1 = lib.chkfile.load(chkfile, 'scf_j1ao/%d' % ia)
        if is_hybrid:
            vk1 = lib.chkfile.load(chkfile, 'scf_k1ao/%d' % ia)
            vjk1 -= 0.5 * hyb * vk1

        v1 = vjk1 + vjk1.transpose(0, 2, 1)
        v1 = v1 + vresp(dm1) + vnuc_deriv(ia)
        dv_ao.append(v1)

    dv_ao = numpy.array(dv_ao).reshape(len(atmlst), -1, 3, nao, nao)
    spin = dv_ao.shape[1]
    assert dv_ao.shape == (len(atmlst), spin, 3, nao, nao)

    dv_ao = dv_ao.transpose(0, 2, 1, 3, 4).reshape(len(atmlst), 3, spin, nao, nao)
    return dv_ao

def gen_vnuc_deriv(mol):
    def func(ia):
        with mol.with_rinv_at_nucleus(ia):
            vrinv  =  mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(ia)
        return vrinv + vrinv.transpose(0, 2, 1)
    return func

# The base for the analytic EPC calculation
class ElectronPhononCouplingBase(eph.mol.eph_fd.ElectronPhononCouplingBase):
    level_shift = 0.0
    conv_tol_cphf = 1e-8
    max_cycle_cphf = 50

    max_cycle = 50
    max_memory = 4000

    def gen_vnuc_deriv(self, mol=None):
        if mol is None: mol = self.mol
        return gen_vnuc_deriv(mol)

    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        raise NotImplementedError

    def solve_mo1(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                        chkfile=None, atmlst=None, verbose=logger.DEBUG):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        if chkfile is None:
            chkfile = self.chkfile

        from mo1 import get_h1ao, solve_mo1
        get_h1ao(
            self.base, mo_energy=mo_energy, 
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            chkfile=chkfile, atmlst=atmlst, 
            verbose=verbose
        )

        assert os.path.exists(chkfile), '%s not found' % chkfile
        solve_mo1(
            self.base, mo_energy=mo_energy, 
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            chkfile=chkfile, conv_tol=self.conv_tol_cphf,
            atmlst=atmlst, max_cycle=self.max_cycle_cphf,
            verbose=verbose
        )
        return chkfile

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        self.dump_flags()
        dv_ao = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, chkfile=self.chkfile,
        )

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

# TODO: implement make_h1, save some intermediate results to chkfile
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)
    
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

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-3)
    # hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        for ia in range(9):
            # print("dv_ref.shape = ", dv_ref.shape)
            # numpy.savetxt(mol.stdout, dv_ref[ia], fmt="% 12.8f", delimiter=", ")

            # print("dv_sol.shape = ", dv_sol.shape)
            # numpy.savetxt(mol.stdout, dv_sol[ia], fmt="% 12.8f", delimiter=", ")

            err = abs(dv_sol[ia] - dv_ref[ia]).max()
            print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))
            # assert err < 1e-6, 
        
        # assert 1  == 2
