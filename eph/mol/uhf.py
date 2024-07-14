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
import eph.mol.eph_fd, eph.mol.rhf
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

# TODO: implement make_h1, save some intermediate results to chkfile
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.uhf.UHF)
        ElectronPhononCouplingBase.__init__(self, method)
    
    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        if scf_obj is None: scf_obj = self.base

        from eph.mol.uks import gen_veff_deriv
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

    mf = scf.UHF(mol)
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

