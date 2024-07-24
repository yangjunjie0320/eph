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

def _fd(scan_obj=None, ix=None, kpts=None, atmlst=None, stepsize=1e-4, v0=None, dm0=None, xyz=None):
    ia, x = atmlst[ix // 3], ix % 3
    
    nao = scan_obj.cell.nao_nr()
    scan_obj.verbose = 10
    p0, p1 = scan_obj.cell.aoslice_by_atom()[ia][2:4]

    spin, nk = dm0.shape[:2]
    spin = dm0.shape[0]
    assert v0.shape == (spin, 3, nk, nao, nao)
    assert kpts.shape == (nk, 3)

    dxyz = numpy.zeros_like(xyz)
    dxyz[ia, x] = stepsize

    c1 = scan_obj.cell.set_geom_(xyz + dxyz, unit="Bohr", inplace=False)
    scan_obj(c1, dm0=dm0[0] if spin == 1 else dm0)
    dm1 = scan_obj.make_rdm1()
    v1  = scan_obj.get_veff(dm=dm1).reshape(spin, nk, nao, nao)
    v1 += scan_obj.get_hcore() - scan_obj.cell.pbc_intor("int1e_kin", kpts=kpts)
    
    c2 = scan_obj.cell.set_geom_(xyz - dxyz, unit="Bohr", inplace=False)
    scan_obj(c2, dm0=dm0[0] if spin == 1 else dm0)
    dm2 = scan_obj.make_rdm1()
    v2  = scan_obj.get_veff(dm=dm2).reshape(spin, nk, nao, nao)
    v2 += scan_obj.get_hcore() - scan_obj.cell.pbc_intor("int1e_kin", kpts=kpts)

    assert v1.shape == v2.shape == (spin, nk, nao, nao)

    dv_ia_x = (v1 - v2) / (2 * stepsize)
    dv_ia_x[:, :, p0:p1, :] -= v0[:, x, :, p0:p1, :]
    dv_ia_x[:, :, :, p0:p1] -= v0[:, x, :, p0:p1, :].transpose(0, 1, 3, 2)
    assert 1 == 2
    return dv_ia_x.reshape(spin, nao, nao)


class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        cell = self.cell
        nao = cell.nao_nr()
        xyz = cell.atom_coords(unit="Bohr")

        if atmlst is None:
            atmlst = range(cell.natm)

        natm = len(atmlst)
        self.dump_flags()

        scf_obj = self.base.to_kscf()
        kpts = scf_obj.kpts
        if not isinstance(kpts, numpy.ndarray):
            kpts = kpts.kpts
        nk = len(kpts)

        from pyscf.pbc.dft.numint import KNumInt
        ni = getattr(scf_obj, '_numint', None)
        if ni is not None and not isinstance(ni, KNumInt):
            scf_obj._numint = KNumInt(kpts)

        scan_obj = scf_obj.as_scanner()
        grad_obj = scf_obj.nuc_grad_method()

        dm0 = scf_obj.make_rdm1()
        dm0 = dm0.reshape(-1, nk, nao, nao)
        spin = dm0.shape[0]
        scf_obj.verbose = 10
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
            dv_ia_x = _fd(
                scan_obj=scan_obj, xyz=xyz,
                ix=ix, atmlst=atmlst, 
                stepsize=stepsize,
                v0=v0, dm0=dm0, kpts=kpts
            )
            dv_ao.append(dv_ia_x)

        # for i0, ia in enumerate(atmlst):
        #     s0, s1, p0, p1 = aoslices[ia]
        #     for x in range(3):
        #         dxyz = numpy.zeros_like(xyz)
        #         dxyz[ia, x] = stepsize

        #         m1 = cell.set_geom_(xyz + dxyz, unit="Bohr", inplace=False)
        #         scan_obj(m1, dm0=dm0[0] if spin == 1 else dm0)
        #         dm1 = scan_obj.make_rdm1()
        #         v1  = scan_obj.get_hcore()
        #         v1 -= numpy.asarray(scan_obj.cell.pbc_intor('int1e_kin', kpts=kpts))
        #         v1  = v1.reshape(1, nk, nao, nao)
        #         v1 += scan_obj.get_veff(dm_kpts=dm1).reshape(spin, nk, nao, nao)
                

        #         scan_obj(xyz - dxyz, dm0=dm0)
        #         dm2 = scan_obj.make_rdm1()
        #         v2  = scan_obj.get_hcore()
        #         v2 -= numpy.asarray(scan_obj.cell.pbc_intor('int1e_kin', kpts=kpts))
        #         v2  = v2.reshape(1, nk, nao, nao)
        #         v2 += scan_obj.get_veff(dm_kpts=dm2).reshape(spin, nk, nao, nao)

        #         assert v1.shape == v2.shape == (spin, nk, nao, nao)

        #         dv_ia_x = (v1 - v2) / (2 * stepsize)

        #         for s in range(spin):
        #             print(dv_ia_x[s, :, p0:p1, :].shape, v0[s, x, :, p0:p1].shape   )
        #             dv_ia_x[s, :, p0:p1, :] -= v0[s, x, :, p0:p1, :]
        #             dv_ia_x[s, :, :, p0:p1] -= v0[s, x, :, p0:p1].transpose(0, 2, 1)

        #         dv_ao.append(dv_ia_x)

        dv_ao = numpy.array(dv_ao).reshape(len(atmlst), 3, spin, nk, nao, nao)
        print(dv_ao.shape)
        assert 1 == 2
        # nao = self.mol.nao_nr()
        # dv_ao = numpy.array(dv_ao).reshape(len(atmlst), 3, spin, nk, nao, nao)
        # self.dv_ao = self._finalize(dv_ao)

        # return self.dv_ao

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
    mf.kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel(stepsize=0.0)
