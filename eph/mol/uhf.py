import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from eph.mol import eph_fd
from eph.mol.eph_fd import harmonic_analysis

class ElectronPhononCoupling(eph.mol.rhf.ElectronPhononCoupling):
    def __init__(self, base, mol, hess=None, dv_ao=None, mass=None,
                 exclude_rot=True, exclude