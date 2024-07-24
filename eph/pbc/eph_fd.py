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

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        if atmlst is None:
            atmlst = range(self.cell.natm)

        self.dump_flags()

        cell = self.cell
        xyz = cell.atom_coords()
        aoslices = cell.aoslice_by_atom()

        scf_obj = self.base.to_kscf()
        kpts = scf_obj.kpts
        if not isinstance(kpts, numpy.ndarray):
            kpts = kpts.kpts
        vk = kpts
        nk = len(vk)

        from pyscf.pbc.dft.numint import KNumInt
        ni = getattr(scf_obj, '_numint', None)
        if ni is not None and not isinstance(ni, KNumInt):
            scf_obj._numint = KNumInt(kpts)

        scf_obj.verbose = 5
        dm0 = scf_obj.make_rdm1()
        scf_obj.kernel(dm0=dm0)

        dm0 = scf_obj.make_rdm1()
        nao = cell.nao_nr()
        dm0 = numpy.asarray(dm0).reshape(-1, nk, nao, nao)
        spin = dm0.shape[0]

        if spin == 1:
            dm0 = dm0[0]

        scan_obj = scf_obj.as_scanner()
        grad_obj = scf_obj.nuc_grad_method()

        from pyscf.pbc.grad import rhf as rhf_grad
        from pyscf.pbc.grad import rks as rks_grad
        veff = grad_obj.get_veff(dm=dm0)
        veff = veff.reshape(spin, 3, nk, nao, nao) # check this

        v1e  = grad_obj.get_hcore()
        v1e -= numpy.asarray(grad_obj.cell.pbc_intor('int1e_ipkin', comp=3, kpts=kpts))
        v1e  = v1e.reshape(nk, 3, 1, nao, nao)
        v0 = veff - numpy.einsum('kxsmn->sxkmn', v1e)
        assert v0.shape == (spin, 3, nk, nao, nao)

        dv_ao = []
        for i0, ia in enumerate(atmlst):
            s0, s1, p0, p1 = aoslices[ia]
            for x in range(3):
                dxyz = numpy.zeros_like(xyz)
                dxyz[ia, x] = stepsize

                scan_obj(xyz + dxyz, inplace=False, unit='B', dm0=dm0)
                dm1 = scan_obj.make_rdm1()
                v1  = scan_obj.get_veff(dm_kpts=dm1).reshape(spin, nk, nao, nao)
                v1 += scan_obj.get_hcore().reshape(1, nk, nao, nao)
                v1 -= numpy.asarray(scan_obj.cell.pbc_intor('int1e_kin', kpts=kpts)).reshape(1, nk, nao, nao)

                scan_obj(xyz - dxyz, dm0=dm0)
                dm2 = scan_obj.make_rdm1()
                v2  = scan_obj.get_veff(dm_kpts=dm2).reshape(spin, nk, nao, nao)
                v2 += scan_obj.get_hcore().reshape(nk, 1, nao, nao)
                v2 -= numpy.asarray(scan_obj.cell.pbc_intor('int1e_kin', kpts=kpts)).reshape(1, nk, nao, nao)

                assert v1.shape == v2.shape == (spin, nk, nao, nao)

                dv_ia_x = (v1 - v2) / (2 * stepsize)

                for s in range(spin):
                    print(dv_ia_x[s, :, p0:p1, :].shape, v0[s, x, :, p0:p1].shape   )
                    dv_ia_x[s, :, p0:p1, :] -= v0[s, x, :, p0:p1, :]
                    dv_ia_x[s, :, :, p0:p1] -= v0[s, x, :, p0:p1].transpose(0, 2, 1)

                dv_ao.append(dv_ia_x)

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
    cell.build()

    from pyscf.pbc import df
    from pyscf.pbc import scf as pbcscf

    mf = pbcscf.RKS(cell)
    # mf.with_df = df.GDF(cell)
    mf.xc = "PBE"
    mf.init_guess = 'atom' # atom guess is fast
    mf.kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()
