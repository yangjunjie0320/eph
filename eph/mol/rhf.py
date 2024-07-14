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
           h1ao=None, mo1=None, atmlst=None,
           max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol
    scf_obj = eph_obj.base

    if mo_energy is None: mo_energy = scf_obj.mo_energy
    if mo_occ    is None: mo_occ    = scf_obj.mo_occ
    if mo_coeff  is None: mo_coeff  = scf_obj.mo_coeff
    if atmlst is None:    atmlst    = range(mol_obj.natm)
    nao, nmo = mo_coeff.shape[-2:]

    if h1ao is None:
        h1ao = eph_obj.make_h1(
            mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, verbose=log
        )
        t1 = log.timer_debug1('making H1', *t0)

    if mo1 is None:
        mo1, mo_e1 = eph_obj.solve_mo1(
            mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            h1ao_or_chkfile=h1ao,
            atmlst=atmlst, verbose=log
        )
        t1 = log.timer_debug1('solving MO1', *t1)

    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol_obj)
    veff_deriv = eph_obj.gen_veff_deriv(
        mo_energy=mo_energy, mo_coeff=mo_coeff,
        mo_occ=mo_occ, mo1=mo1, h1ao=h1ao, verbose=log
        )
    
    dv_ao = [] # numpy.zeros((len(atmlst), 3, nao, nao))
    for i0, ia in enumerate(atmlst):
        v = vnuc_deriv(ia) + veff_deriv(ia)
        dv_ao.append(v)

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
    max_cycle = 50
    max_memory = 4000

    def gen_vnuc_deriv(self, mol=None):
        if mol is None: mol = self.mol
        return gen_vnuc_deriv(mol)

    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        raise NotImplementedError

    def make_h1(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                      chkfile=None, atmlst=None, verbose=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        res = self.base.Hessian().make_h1(
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            chkfile=chkfile, atmlst=atmlst, 
            verbose=verbose
        )
        return res
    
    def solve_mo1(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                        h1ao_or_chkfile=None, atmlst=None, verbose=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        res = self.base.Hessian().solve_mo1(
            mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            h1ao_or_chkfile=h1ao_or_chkfile, atmlst=atmlst,
            verbose=verbose, max_memory=self.max_memory,
        )
        return res

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        self.dump_flags()
        dv_ao = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, h1ao=None, mo1=None,
        )

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

# TODO: implement make_h1, save some intermediate results to chkfile
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        if scf_obj is None: scf_obj = self.base
        
        # use the function within eph.mol.rks,
        # treat as a hybrid functional with hyb = 1.0
        from eph.mol.rks import gen_veff_deriv
        res = gen_veff_deriv(
            mo_occ=mo_occ, mo_coeff=mo_coeff, scf_obj=scf_obj,
            mo1=mo1, h1ao=h1ao, verbose=verbose
            )
        
        return res
    
if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       0.0000000000     0.0000000000     0.1146878262
    H      -0.7540663886    -0.0000000000    -0.4587203947
    H       0.7540663886    -0.0000000000    -0.4587203947
    '''
    mol.basis = '631g*'
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
