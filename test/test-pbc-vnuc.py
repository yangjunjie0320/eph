import os, sys
import numpy, scipy

import pyscf
from pyscf.pbc import gto, scf

import eph
from eph.pbc.rhf import ElectronPhononCoupling

from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
c = bulk("C", "diamond", a=3.5668)

from pyscf.pbc import gto, scf
cell = gto.Cell()
cell.atom = ase_atoms_to_pyscf(c)
cell.a = c.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.unit = 'A'
cell.verbose = 0
cell.ke_cutoff = 100
cell.exp_to_discard = 0.1
cell.build()

natm = cell.natm
nao = cell.nao_nr()

mf = scf.RHF(cell)
mf.verbose = 0
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.max_cycle = 100
mf.exxdiv = None
mf.kernel(dm0=None)
dm0 = mf.make_rdm1()

eph_obj = ElectronPhononCoupling(mf)
grad_obj = mf.to_kscf().nuc_grad_method()

h1 = -eph_obj._hcore_deriv()
h2 = grad_obj.get_hcore()[0]
assert numpy.allclose(h1, h2)

func1 = eph_obj.gen_hcore_deriv()
func2 = grad_obj.hcore_generator()

v1 = numpy.asarray([func1(ia) for ia in range(natm)]).reshape(-1, nao, nao)
v2 = numpy.asarray([func2(ia) for ia in range(natm)]).reshape(-1, nao, nao)

for ix in range(natm * 3):
    v1x = v1[ix] # * 0.5
    v2x = v2[ix]

    err = abs(v1x - v2x).max()
    if err < 1e-6:
        continue

    v1x = v1x[:10, :10]
    v2x = v2x[:10, :10]

    numpy.set_printoptions(precision=4, linewidth=150)
    print("Error:", err)
    print("Analytical:")
    print(v1x)
    print("Finite difference:")
    print(v2x)