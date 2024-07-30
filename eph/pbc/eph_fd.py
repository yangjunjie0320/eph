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

    def _finalize(self, dv_ao):
        assert dv_ao is not None
        if not isinstance(dv_ao, numpy.ndarray):
            spin = dv_ao[0].shape[0]
            nao = dv_ao[0].shape[-1]
            nk = dv_ao[0].shape[1]

            dv_ao = numpy.asarray(dv_ao)
            dv_ao = dv_ao.reshape(-1, 3, spin, nk, nao, nao)

        natm, _, spin, nk, nao, _ = dv_ao.shape
        assert dv_ao.shape == (natm, 3, spin, nk, nao, nao)

        if spin == 1:
            dv_ao = dv_ao.reshape(-1, nk, nao, nao)

        elif spin == 2:
            dv_ao = dv_ao.reshape(-1, 2, nk, nao, nao)
            dv_ao = numpy.asarray((dv_ao[:, 0], dv_ao[:, 1]))
            assert dv_ao.shape == (spin, natm * 3, nk, nao, nao)

        else:
            raise RuntimeError("spin = %d is not supported" % spin)

        return dv_ao

def _fd(scf_obj=None, ix=None, atmlst=None, stepsize=1e-4, v0=None, dm0=None):
    ia, x = atmlst[ix // 3], ix % 3

    cell = scf_obj.cell
    scan_obj = scf_obj.as_scanner()

    stdout = scf_obj.stdout
    s = cell.aoslice_by_atom()
    nao = s[-1][-1]
    p0, p1 = s[ia][2:]

    kpts = scf_obj.kpts
    if not isinstance(kpts, numpy.ndarray):
        kpts = kpts.kpts
    nk = len(kpts)

    dm0 = dm0.reshape(-1, nk, nao, nao)
    spin, nk = dm0.shape[:2]
    assert v0.shape == (spin, 3, nk, nao, nao)
    assert kpts.shape == (nk, 3)

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

    c1 = cell.set_geom_(xyz + dxyz, unit="Bohr", inplace=False)
    c1.a = cell.lattice_vectors()
    c1.unit = "Bohr"
    c1.build()

    scan_obj(c1, dm0=dm0[0] if spin == 1 else dm0)
    dm1 = scan_obj.make_rdm1()
    v1  = scan_obj.get_veff(dm=dm1).reshape(spin, nk, nao, nao)
    v1 += scan_obj.get_hcore() - c1.pbc_intor('int1e_kin', kpts=kpts)

    c2 = cell.set_geom_(xyz - dxyz, unit="Bohr", inplace=False)
    c2.a = cell.lattice_vectors()
    c2.unit = "Bohr"
    c2.build()

    scan_obj(c2, dm0=dm0[0] if spin == 1 else dm0)
    dm2 = scan_obj.make_rdm1()
    v2  = scan_obj.get_veff(dm=dm2).reshape(spin, nk, nao, nao)
    v2 += scan_obj.get_hcore() - c2.pbc_intor('int1e_kin', kpts=kpts)

    assert v1.shape == v2.shape == (spin, nk, nao, nao)
    dv = (v1 - v2) / (2 * stepsize)
    dv[:, :, p0:p1, :] -= v0[:, x, :, p0:p1, :]
    dv[:, :, :, p0:p1] -= v0[:, x, :, p0:p1, :].transpose(0, 1, 3, 2).conj()
    return dv

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        cell = self.cell
        nao = cell.nao_nr()

        if atmlst is None:
            atmlst = range(cell.natm)

        natm = len(atmlst)
        self.dump_flags()

        scf_obj = self.base.to_kscf()
        grad_obj = scf_obj.nuc_grad_method()

        kpts = scf_obj.kpts
        if not isinstance(kpts, numpy.ndarray):
            kpts = kpts.kpts
        nk = len(kpts)

        from pyscf.pbc.dft.numint import KNumInt
        ni = getattr(scf_obj, '_numint', None)
        if ni is not None and not isinstance(ni, KNumInt):
            scf_obj._numint = KNumInt(kpts)

        dm0 = scf_obj.make_rdm1()
        dm0 = dm0.reshape(-1, nk, nao, nao)
        spin = dm0.shape[0]
        assert scf_obj.converged

        v0 = grad_obj.get_hcore()
        v0 -= numpy.asarray(cell.pbc_intor("int1e_ipkin", kpts=kpts))
        v0 = v0.transpose(1, 0, 2, 3)
        v0 = grad_obj.get_veff().reshape(3, spin, nk, nao, nao) - v0[:, None]
        v0 = v0.transpose(1, 0, 2, 3, 4)

        dv_ao = []
        for ix in range(3 * natm):
            dv_ao_ia_x = _fd(
                scf_obj=scf_obj, ix=ix, atmlst=atmlst,
                stepsize=stepsize, v0=v0, dm0=dm0
                )
            dv_ao.append(dv_ao_ia_x)

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 0
    cell.mesh = [10, 10, 10]
    cell.build()

    mf = scf.UKS(cell)
    mf.xc = "PBE0"
    mf.init_guess = 'atom'
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    stepsize = 1e-4
    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel(stepsize=stepsize / 2)

    from pyscf.pbc.eph.eph_fd import gen_cells, run_mfs, get_vmat
    mf = mf.to_kscf()
    cells_a, cells_b = gen_cells(cell, stepsize / 2.0)
    mfset = run_mfs(mf, cells_a, cells_b) # run mean field calculations on all these cells
    dv_ref = get_vmat(mf, mfset, stepsize) # extracting <u|dV|v>/dR
    spin = 2 if dv_ref.ndim == 4 else 1
    
    nao = cell.nao_nr()
    if spin == 2:
        dv_ref = dv_ref.transpose(1, 0, 2, 3)
        dv_ref = dv_ref.reshape(-1, nao, nao)
        dv_sol = dv_sol.reshape(-1, nao, nao)

    import sys
    for n in range(dv_sol.shape[0]):
        err = abs(dv_sol[n] - dv_ref[n]).max()
        if err > 1e-4:
            print("n = %d, error = % 6.4e" % (n, abs(dv_sol[n] - dv_ref[n]).max()))
            print("dv_sol = ")
            numpy.savetxt(sys.stdout, dv_sol[n], fmt="% 8.4f", delimiter=", ")

            print("dv_ref = ")
            numpy.savetxt(sys.stdout, dv_ref[n], fmt="% 8.4f", delimiter=", ")

    err = abs(dv_sol - dv_ref).max()
    print("error = % 6.4e" % err)