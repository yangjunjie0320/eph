import os, numpy, scipy, tempfile

import eph.mol
import eph.mol.eph_fd
import pyscf
from pyscf import lib, scf
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
import pyscf.pbc
from pyscf.scf import hf, _vhf
from pyscf import hessian

from pyscf.pbc.scf.khf import KSCF
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.multigrid import MultiGridFFTDF2

import eph
from eph.pbc import eph_fd

def kernel(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None):
    pass

def make_h1(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None, 
            atmlst=None, verbose=None):
    atmlst = atmlst if not atmlst else range(cell.natm)

    cell = eph_obj.cell
    ao_slices = cell.aoslice_by_atom()
    nao, nmo = mo_coeff.shape
    nbas = cell.nbas

    assert mo_coeff.shape == (nao, nmo)
    assert mo_energy.shape == (nmo, )
    assert mo_occ.shape == (nmo, )

    orbo = mo_coeff[:, mo_occ > 0]
    nocc = orbo.shape[1]
    dm0 = numpy.dot(orbo, orbo.T.conj()) * 2
    assert isinstance(eph_obj.base, pyscf.pbc.scf.hf.RHF)

    from pyscf.pbc.grad.krhf import hcore_generator
    hcore_deriv = hcore_generator(cell, kpts=numpy.zeros(3).reshape(-1, 3))

    for i0, ia in enumerate(atmlst):
        s0, s1, p0, p1 = cell.aoslice_by_atom()[ia]
        shls_slice  = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s2kl', -dm0[:, p0:p1]] # vj1
        script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1

        vj1, vj2, vk1, 

def solve_mo1(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None,
                h1ao_or_chkfile=None, atmlst=None, verbose=None):
    pass

# The base for the analytic EPC calculation
class ElectronPhononCouplingBase(eph.pbc.eph_fd.ElectronPhononCouplingBase):
    level_shift = 0.0
    max_cycle = 50
    max_memory = 4000

    def __init__(self, method):
        assert isinstance(method, pyscf.pbc.scf.hf.KRHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def gen_vnuc_deriv(self, cell=None):
        if cell is None: cell = self.cell
        from eph.mol.rhf import gen_vnuc_deriv
        return gen_vnuc_deriv(cell)

    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        raise NotImplementedError

    def make_h1(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                      chkfile=None, atmlst=None, verbose=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        res = self.base.Hessian().make_h1(
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            chkfile=chkfile, atmlst=atmlst, 
            verbose=verbose
        )
        return res
    
    def solve_mo1(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                        h1ao_or_chkfile=None, atmlst=None, verbose=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        res = self.base.Hessian().solve_mo1(
            mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            h1ao_or_chkfile=h1ao_or_chkfile, atmlst=atmlst,
            verbose=verbose, max_memory=self.max_memory,
        )
        return res

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc.dft import multigrid

    cell = gto.Cell()
    cell.atom = '''
    C 0.0000  0.0000  0.0000
    C 0.8917  0.8917  0.8917
    C 1.7834  1.7834  0.0000
    C 2.6751  2.6751  0.8917
    C 1.7834  0.0000  1.7834
    C 2.6751  0.8917  2.6751
    C 0.0000  1.7834  1.7834
    C 0.8917  2.6751  2.6751'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = numpy.eye(3) * 3.5668
    cell.unit = 'A'
    cell.verbose = 4
    cell.ke_cutoff = 200
    cell.build()

    stepsize = 1e-4
    mf = scf.RKS(cell)
    mf.xc = "PBE"
    mf.with_df = multigrid.MultiGridFFTDF2(cell)
    mf.init_guess = 'atom'
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel(dm0=None)
    dm0 = mf.make_rdm1()

    eph_obj = ElectronPhononCoupling(mf)
    dv_1 = eph_obj.kernel(stepsize=stepsize, atmlst=[0])

    mf = scf.RKS(cell)
    mf.xc = "PBE"
    mf.init_guess = 'atom'
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel()
    eph_obj = ElectronPhononCoupling(mf)
    dv_2 = eph_obj.kernel(stepsize=stepsize, atmlst=[0])

    err = abs(dv_1 - dv_2).max()
    print("stepsize = % 6.2e, error = % 6.2e" % (stepsize, err))


    # from pyscf.pbc.eph.eph_fd import gen_cells, run_mfs, get_vmat
    # cells_a, cells_b = gen_cells(cell, stepsize / 2.0)
    # mfk = mf.to_kscf()
    # mfset = run_mfs(mfk, cells_a[:3], cells_b[:3])
    # dv_ref = get_vmat(mfk, mfset, stepsize) 

    # eph_obj = ElectronPhononCoupling(mf)
    # dv_sol  = eph_obj.kernel(stepsize=stepsize / 2.0, atmlst=[0])
    # dv_ref = dv_ref.reshape(dv_sol.shape)

    # err = abs(dv_sol - dv_ref).max()
    # print("stepsize = % 6.2e, error = % 6.2e" % (stepsize, err))
