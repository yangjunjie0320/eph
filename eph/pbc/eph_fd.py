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

def _fd(scf_obj=None, ix=None, atmlst=None, stepsize=1e-4, v0=None, dm0=None):
    ia, x = atmlst[ix // 3], ix % 3

    cell = scf_obj.cell
    scan_obj = scf_obj.as_scanner()

    stdout = scf_obj.stdout
    s = cell.aoslice_by_atom()
    nao = s[-1][-1]
    p0, p1 = s[ia][2:]

    spin = dm0.shape[0]
    assert v0.shape == (spin, 3, nao, nao)

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

    scan_obj(c1, dm0=dm0)
    dm1 = scan_obj.make_rdm1()
    v1  = scan_obj.get_veff(dm=dm1).reshape(spin, nao, nao)
    v1 += scan_obj.get_hcore() - c1.pbc_intor('int1e_kin')

    c2 = cell.set_geom_(xyz - dxyz, unit="Bohr", inplace=False)
    c2.a = cell.lattice_vectors()
    c2.unit = "Bohr"
    c2.build()

    scan_obj(c2, dm0=dm0)
    dm2 = scan_obj.make_rdm1()
    v2  = scan_obj.get_veff(dm=dm2).reshape(spin, nao, nao)
    v2 += scan_obj.get_hcore() - c2.pbc_intor('int1e_kin')

    assert v1.shape == v2.shape == (spin, nao, nao)
    dv = (v1 - v2) / (2 * stepsize)
    dv[:, p0:p1, :] -= v0[:, x, p0:p1, :]
    dv[:, :, p0:p1] -= v0[:, x, p0:p1, :].transpose(0, 2, 1)
    return dv

import pyscf.pbc.grad
pyscf.pbc.scf.hf.RHF.nuc_grad_method  = lambda self: pyscf.pbc.grad.rhf.Gradients(self)
pyscf.pbc.scf.uhf.UHF.nuc_grad_method = lambda self: pyscf.pbc.grad.uhf.Gradients(self)
pyscf.pbc.dft.rks.RKS.nuc_grad_method = lambda self: pyscf.pbc.grad.rks.Gradients(self)
pyscf.pbc.dft.uks.UKS.nuc_grad_method = lambda self: pyscf.pbc.grad.uks.Gradients(self)

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        cell = self.cell
        nao = cell.nao_nr()

        if atmlst is None:
            atmlst = range(cell.natm)

        natm = len(atmlst)
        self.dump_flags()

        assert not isinstance(self.base, pyscf.pbc.scf.khf.KSCF)
        if isinstance(self.base.with_df, pyscf.pbc.dft.multigrid.MultiGridFFTDF2):
            scf_obj = self.base
            grad_obj = scf_obj.nuc_grad_method()
            assert scf_obj.converged

            dm0 = scf_obj.make_rdm1()
            dm0 = dm0.reshape(-1, nao, nao)
            spin = dm0.shape[0]
            dm0 = dm0[0] if spin == 1 else dm0

            v0 = grad_obj.get_veff(dm=dm0)
            print(v0.shape)
            v0  = v0.reshape(3, spin, nao, nao)
            v0 -= grad_obj.get_hcore().reshape(3, -1, nao, nao)
            v0 += cell.pbc_intor("int1e_ipkin").reshape(3, -1, nao, nao)
            v0  = v0.transpose(1, 0, 2, 3)
            assert v0.shape == (spin, 3, nao, nao)

        elif isinstance(self.base.with_df, pyscf.pbc.df.FFTDF):
            scf_obj = self.base.to_kscf()
            grad_obj = scf_obj.nuc_grad_method()
            assert scf_obj.converged

            dm0 = scf_obj.make_rdm1()
            dm0 = dm0.reshape(-1, 1, nao, nao)
            spin = dm0.shape[0]
            dm0 = dm0[0] if spin == 1 else dm0

            v0 = grad_obj.get_hcore()
            v0 -= numpy.asarray(cell.pbc_intor("int1e_ipkin", kpts=scf_obj.kpts))
            v0 = v0.transpose(1, 0, 2, 3)
            v0 = grad_obj.get_veff().reshape(3, spin, 1, nao, nao) - v0[:, None]
            v0 = v0.transpose(1, 0, 2, 3, 4).reshape(spin, 3, nao, nao)

        else:
            raise NotImplementedError
    
        dv_ao = []
        for ix in range(3 * natm):
            print(dm0.shape)
            dv_ao_ia_x = _fd(
                scf_obj=scf_obj, ix=ix, atmlst=atmlst, 
                stepsize=stepsize, v0=v0, dm0=dm0
                )
            dv_ao.append(dv_ao_ia_x)

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft import multigrid

    cell = gto.Cell()
    cell.atom = '''
    Li 2.00000 2.000000 2.00000
    Li 2.00000 2.000000 4.00000
    '''
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.a = numpy.diag([4, 4, 6])
    cell.ke_cutoff = 200
    cell.precision = 1e-6
    cell.verbose = 0
    cell.use_loose_rcut = True
    cell.use_particle_mesh_ewald = True
    cell.exp_to_discard = 0.1
    cell.build()

    mf = scf.RKS(cell)
    # mf.with_df = multigrid.MultiGridFFTDF2(cell)
    # mf.with_df.ngrids = 4
    mf.xc = "PBE"
    mf.init_guess = 'atom'
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel()

    stepsize = 1e-4
    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel(stepsize=stepsize/2.0)

    from pyscf.pbc.eph.eph_fd import gen_cells, run_mfs, get_vmat
    # mf.with_df = pyscf.pbc.df.FFTDF(cell)
    mf = mf.to_kscf()
    cells_a, cells_b = gen_cells(cell, stepsize / 2.0)
    mf.verbose = 4
    mfset = run_mfs(mf, cells_a, cells_b) # run mean field calculations on all these cells

    dv_ref = get_vmat(mf, mfset, stepsize) # extracting <u|dV|v>/dR
    dv_ref = dv_ref.reshape(dv_sol.shape)

    for n in range(dv_sol.shape[0]):
        err = abs(dv_sol[n] - dv_ref[n]).max()
        import sys
        print("\nn = %d, error = % 6.4e" % (n, abs(dv_sol[n] - dv_ref[n]).max()))
        print("dv_sol:")
        numpy.savetxt(sys.stdout, dv_sol[n], fmt="% 8.4f", delimiter=", ")
        print("dv_ref:")
        numpy.savetxt(sys.stdout, dv_ref[n], fmt="% 8.4f", delimiter=", ")

    err = abs(dv_sol - dv_ref).max()
    print("error = % 6.4e" % err)