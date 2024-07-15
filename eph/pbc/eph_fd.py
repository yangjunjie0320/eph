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
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

class ElectronPhononCoupling(eph.mol.eph_fd.ElectronPhononCoupling):
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

        v0 = grad_obj.get_veff(dm=dm0) + grad_obj.get_hcore() + self.base.mol.intor("int1e_ipkin")
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

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    dv_fd = eph_fd.kernel(stepsize=1e-4)

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    dv_fd = eph_fd.kernel(stepsize=1e-4)

    mf = scf.RKS(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.xc = "LDA"
    mf.kernel()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    dv_fd = eph_fd.kernel(stepsize=1e-4)

    mf = scf.UKS(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.xc = "LDA"
    mf.kernel()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    dv_fd = eph_fd.kernel(stepsize=1e-4)
    