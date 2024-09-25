import os, numpy, scipy, tempfile

import eph.mol
import eph.mol.eph_fd
import pyscf
from pyscf import lib, scf
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from pyscf.pbc.scf.khf import KSCF
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.multigrid import MultiGridFFTDF2

import eph
from eph.mol import eph_fd, uhf

# PBC electron-phonon coupling base class
class ElectronPhononCouplingBase(eph.mol.eph_fd.ElectronPhononCouplingBase):
    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.chkfile = method.chkfile

        self.mol = self.cell = method.mol
        self.base = method
        self.atmlst = None

        self.max_memory = method.max_memory
        self.unit = 'au'
        self.dv_ao = None

def _fd(scf_obj=None, ix=None, atmlst=None, stepsize=1e-4, v0=None, dm0=None):
    # the finite difference will only be performed inside
    # the atmlst
    ia, x = atmlst[ix // 3], ix % 3

    cell = scf_obj.cell
    nao = cell.nao_nr()

    stdout = scf_obj.stdout
    s = cell.aoslice_by_atom()
    nao = s[-1][-1]
    p0, p1 = s[ia][2:]

    xyz = cell.atom_coords(unit="Bohr")
    dxyz = numpy.zeros_like(xyz)
    dxyz[ia, x] = stepsize

    if scf_obj.verbose >= logger.DEBUG:
        info = "Finite difference for atom %d, direction %d" % (ia, x)
        stdout.write("\n" + "*" * len(info) + "\n")
        stdout.write(info + "\n")
        stdout.write("\noriginal geometry (bohr):\n")
        numpy.savetxt(stdout, xyz, fmt="% 12.8f", delimiter=", ")
        stdout.write("\nperturbed geometry 1 (bohr):\n")
        numpy.savetxt(stdout, xyz + dxyz, fmt="% 12.8f", delimiter=", ")
        stdout.write("\nperturbed geometry 2 (bohr):\n")
        numpy.savetxt(stdout, xyz - dxyz, fmt="% 12.8f", delimiter=", ")

    dm0 = dm0.reshape(scf_obj.make_rdm1().shape)

    def build(xyz):
        c = cell.copy(deep=True)
        c.set_geom_(xyz, unit="Bohr", inplace=True)
        c.a = cell.lattice_vectors()
        c.unit = "Bohr"
        c.build()
        
        s = scf_obj.copy()
        s.reset(c)
        return c, s

    c1, s1 = build(xyz + dxyz)
    s1.kernel(dm0=dm0)
    v1 = s1.get_veff(dm=s1.make_rdm1()).reshape(nao, nao)
    v1 += s1.get_hcore() - c1.pbc_intor('int1e_kin')

    c2, s2 = build(xyz - dxyz)
    s2.kernel(dm0=dm0)
    v2  = s2.get_veff(dm=s2.make_rdm1()).reshape(nao, nao)
    v2 += s2.get_hcore() - c2.pbc_intor('int1e_kin')

    assert v1.shape == v2.shape == (nao, nao)
    return (v1 - v2) / (2 * stepsize)

import pyscf.pbc.grad
pyscf.pbc.scf.hf.RHF.nuc_grad_method  = lambda self: pyscf.pbc.grad.rhf.Gradients(self)
pyscf.pbc.scf.uhf.UHF.nuc_grad_method = lambda self: pyscf.pbc.grad.uhf.Gradients(self)
pyscf.pbc.dft.rks.RKS.nuc_grad_method = lambda self: pyscf.pbc.grad.rks.Gradients(self)
pyscf.pbc.dft.uks.UKS.nuc_grad_method = lambda self: pyscf.pbc.grad.uks.Gradients(self)

# Gamma-point electron-phonon coupling for PBC
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        if atmlst is None:
            atmlst = range(self.cell.natm)

        mf = self.base # .to_kscf()
        cell_obj = mf.cell
        nao = cell_obj.nao_nr()
        natm = len(atmlst)

        kscf_obj = mf.to_kscf()
        dm0 = kscf_obj.make_rdm1()
        kpts = mf.kpts

        grad_obj = kscf_obj.nuc_grad_method()

        veff1 = grad_obj.get_veff()
        assert veff1.shape == (3, 1, nao, nao)

        v1e  = grad_obj.get_hcore()
        v1e -= numpy.asarray(cell_obj.pbc_intor("int1e_ipkin", kpts=kpts))
        assert v1e.shape == (1, 3, nao, nao)

        v0 = veff1 - v1e.transpose(1, 0, 2, 3)
        assert v0.shape == (3, 1, nao, nao)

        dv_ao = []
        for ix in range(3 * natm):
            ia, x = divmod(ix, 3)
            p0, p1 = cell_obj.aoslice_by_atom()[ia][2:]

            dv = _fd(
                scf_obj=self.base, ix=ix, 
                atmlst=atmlst, stepsize=stepsize, 
                dm0=dm0
                )
            dv[:, p0:p1, :] -= v0[x, :, p0:p1]
            dv[:, :, p0:p1] -= v0[x, :, p0:p1].transpose(0, 2, 1).conj()
            assert dv.shape == (1, nao, nao)

            dv_ao.append(dv)

        dv_ao = numpy.asarray(dv_ao).reshape(natm * 3, nao, nao)
        return dv_ao

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft import multigrid

    cell = gto.Cell()
    cell.atom = '''
    Li 1.000000 1.000000 1.000000
    Li 1.000000 1.000000 2.000000
    '''
    cell.a = numpy.diag([2.0, 2.0, 3.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.unit = 'A'
    cell.verbose = 0
    cell.ke_cutoff = 100
    cell.exp_to_discard = 0.1
    cell.build()

    stepsize = 1e-4
    mf = scf.RHF(cell)
    # mf.xc = "PBE"
    mf.verbose = 0
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel()
    dm0 = mf.make_rdm1()

    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.verbose = 0
    dv_sol = eph_obj.kernel(stepsize=stepsize/2.0)
    
    mf = scf.RHF(cell)
    # mf.xc = "PBE"
    mf.verbose = 0
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel(dm0=dm0)

    from pyscf.pbc.eph.eph_fd import gen_cells, run_mfs, get_vmat
    cells_a, cells_b = gen_cells(cell, stepsize/2.0)
    mfk = mf.to_kscf()
    mfset = run_mfs(mfk, cells_a, cells_b)
    dv_ref = get_vmat(mfk, mfset, stepsize) 
    dv_ref = dv_ref.reshape(dv_sol.shape)

    err = abs(dv_sol - dv_ref).max()
    print("stepsize = % 6.2e, error = % 6.2e" % (stepsize, err))

    for x in range(3 * cell.natm):
        err = abs(dv_sol[x] - dv_ref[x]).max()
        
        print(f"\nix = {x}, error = {err:6.4e}")
        print(f"dv_sol[{x}] = ")
        numpy.savetxt(mf.stdout, dv_sol[x], fmt="% 6.4e", delimiter=", ")

        print(f"dv_ref[{x}] = ")
        numpy.savetxt(mf.stdout, dv_ref[x], fmt="% 6.4e", delimiter=", ")
