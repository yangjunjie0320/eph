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

# fix the missing nuc_grad_method
from pyscf.pbc import grad
pyscf.pbc.scf.hf.RHF.nuc_grad_method  = lambda self: pyscf.pbc.grad.rhf.Gradients(self)
pyscf.pbc.scf.uhf.UHF.nuc_grad_method = lambda self: pyscf.pbc.grad.uhf.Gradients(self)

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        if atmlst is None:
            atmlst = range(self.mol.natm)

        self.dump_flags()

        mol = self.mol
        xyz = mol.atom_coords()
        aoslices = mol.aoslice_by_atom()

        dm0 = self.base.make_rdm1()
        nao = mol.nao_nr()
        dm0 = dm0.reshape(-1, nao, nao)
        spin = dm0.shape[0]
        if spin == 1:
            dm0 = dm0[0]

        scan_obj = self.base.as_scanner()
        grad_obj = self.base.nuc_grad_method()

        v0 = grad_obj.get_veff(dm=dm0) + grad_obj.get_hcore() + self.base.cell.intor("int1e_ipkin")
        v0 = v0.reshape(spin, 3, nao, nao)

        dv_ao = []
        for i0, ia in enumerate(atmlst):
            s0, s1, p0, p1 = aoslices[ia]
            for x in range(3):
                dxyz = numpy.zeros_like(xyz)
                dxyz[ia, x] = stepsize

                scan_obj(mol.set_geom_(xyz + dxyz, inplace=False, unit='B'), dm0=dm0)
                dm1 = scan_obj.make_rdm1()
                v1  = scan_obj.get_veff(dm=dm1).reshape(spin, nao, nao)
                v1 += scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")
                

                scan_obj(mol.set_geom_(xyz - dxyz, inplace=False, unit='B'), dm0=dm0)
                dm2 = scan_obj.make_rdm1()
                v2  = scan_obj.get_veff(dm=dm2).reshape(spin, nao, nao)
                v2 += scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")

                assert v1.shape == v2.shape == (spin, nao, nao)

                dv_ia_x = (v1 - v2) / (2 * stepsize)

                for s in range(spin):
                    dv_ia_x[s, p0:p1, :] -= v0[s, x, p0:p1]
                    dv_ia_x[s, :, p0:p1] -= v0[s, x, p0:p1].T

                dv_ao.append(dv_ia_x)

        nao = self.mol.nao_nr()
        dv_ao = numpy.array(dv_ao).reshape(len(atmlst), 3, spin, nao, nao)
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
    cell.a = diamond.cell
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(diamond)
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.ke_cutoff = 200
    cell.precision = 1e-6
    cell.verbose = 5
    cell.build()

    from pyscf.pbc import dft as pbcdft
    from pyscf.pbc.dft import multigrid

    mf = pbcdft.RKS(cell)
    #mf.xc = "LDA, VWN"
    mf.xc = "PBE,PBE"
    mf.init_guess = 'atom' # atom guess is fast
    mf.with_df = multigrid.MultiGridFFTDF2(cell)
    mf.with_df.ngrids = 4 # number of sets of grid points
    mf.kernel()

    from pyscf.pbc.grad import rks as rks_grad
    grad = rks_grad.Gradients(mf)
    g = grad.kernel()
