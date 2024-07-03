import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from eph.mol import eph_fd, rhf
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

class ElectronPhononCoupling(eph.mol.rhf.ElectronPhononCouplingBase):
    def __init__(self, base, mol):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
        if scf_obj is None: scf_obj = self.base
        if mo1 is None: mo1 = self.solve_mo1(mo_coeff, mo_occ, h1ao, log)
        return self.gen_veff_deriv_(mo_occ, mo_coeff, scf_obj, mo1, h1ao, log)
    
    def make_h1(self, mo_coeff, mo_occ, tmpfile=None, atmlst=None, log=None):
        pass

