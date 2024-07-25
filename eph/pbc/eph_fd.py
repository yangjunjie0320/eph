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
    dxyz = dxyz.reshape(-1, 3)

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
    s1 = scf_obj.__class__(c1)
    s1.kpts = kpts
    if hasattr(scf_obj, 'with_df'):
        s1.with_df = scf_obj.with_df.reset(cell=c1)

    if hasattr(scf_obj, 'xc'):
        s1.xc = scf_obj.xc
    
    s1.exxdiv = getattr(scf_obj, 'exxdiv', None)
    s1.conv_tol = scf_obj.conv_tol
    s1.conv_tol_grad = scf_obj.conv_tol_grad
    s1.max_cycle = scf_obj.max_cycle
    s1.verbose = scf_obj.verbose
    s1.kernel(dm0=dm0[0] if spin == 1 else dm0)

    dm1 = s1.make_rdm1()
    v1 = s1.get_veff(dm=dm1).reshape(spin, nk, nao, nao)
    v1 += s1.get_hcore() - c1.pbc_intor('int1e_kin', kpts=kpts)

    c2 = cell.set_geom_(xyz - dxyz, unit="Bohr", inplace=False)
    s2 = scf_obj.__class__(c2)
    s2.kpts = kpts
    if hasattr(scf_obj, 'with_df'):
        s2.with_df = scf_obj.with_df.reset(cell=c2)

    if hasattr(scf_obj, 'xc'):
        s2.xc = scf_obj.xc
    
    s2.exxdiv = getattr(scf_obj, 'exxdiv', None)
    s2.conv_tol = scf_obj.conv_tol
    s2.conv_tol_grad = scf_obj.conv_tol_grad
    s2.max_cycle = scf_obj.max_cycle
    s2.verbose = scf_obj.verbose
    s2.kernel(dm0=dm0[0] if spin == 1 else dm0)

    dm2 = s2.make_rdm1()
    v2 = s2.get_veff(dm=dm2).reshape(spin, nk, nao, nao)
    v2 += s2.get_hcore() - c2.pbc_intor('int1e_kin', kpts=kpts)

    assert v1.shape == v2.shape == (spin, nk, nao, nao)
    dv = (v1 - v2) / (2 * stepsize)
    dv[:, :, p0:p1, :] -= v0[:, x, :, p0:p1, :]
    dv[:, :, :, p0:p1] -= v0[:, x, :, p0:p1, :].transpose(0, 1, 3, 2)
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
        scf_obj.kernel(dm0=dm0[0] if spin == 1 else dm0)
        assert scf_obj.converged

        dm0 = scf_obj.make_rdm1().reshape(spin, nk, nao, nao)
        v0 = grad_obj.get_veff(dm=dm0[0] if spin == 1 else dm0)
        v0 = v0.reshape(spin, 3, nk, nao, nao)
        v0 += grad_obj.get_hcore().transpose(1, 0, 2, 3)
        v0 += numpy.asarray(grad_obj.cell.pbc_intor('int1e_ipkin', kpts=kpts, comp=3)).transpose(1, 0, 2, 3)
        assert v0.shape == (spin, 3, nk, nao, nao)

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
    from pyscf.pbc.tools import pyscf_ase

    import ase
    import ase.lattice
    from ase.lattice.cubic import Diamond
    diamond = Diamond(symbol='C', latticeconstant=3.5668)

    cell = gto.Cell()
    cell.a = numpy.diag([2, 2, 6])
    cell.atom = """
    He 1.000000 1.000000 2.000000
    He 1.000000 1.000000 4.000000
    """
    cell.basis  = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [10] * 3
    cell.build()

    from pyscf.pbc import df
    from pyscf.pbc import scf as pbcscf

    mf = pbcscf.RKS(cell)
    # mf.with_df = df.GDF(cell)
    mf.xc = "PBE"
    mf.init_guess = 'atom' # atom guess is fast
    mf.verbose = 0
    mf.kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel(stepsize=1e-4)

    # from pyscf.pbc.eph.eph_fd import gen_cells, run_mfs
    # disp = 1e-2

    # mf = mf.to_kscf()
    # cell = mf.cell
    # cells_a, cells_b = gen_cells(cell, disp/2.0) # generate a bunch of cells with disp/2 on each cartesian coord
    # mf.verbose = 10
    # mfset = run_mfs(mf, cells_a, cells_b) # run mean field calculations on all these cells
    # # vmat = get_vmat(mf, mfset, disp) # extracting <u|dV|v>/dR
    # # hmat = run_hess(mfset, disp)
    # # omega, vec = solve_hmat(cell, hmat)
    # # mass = cell.atom_mass_list() * MP_ME