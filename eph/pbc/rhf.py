import os, numpy, scipy, tempfile

import eph.mol
import eph.mol.eph_fd
import eph.mol.rhf
import pyscf
from pyscf import lib, scf
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
import pyscf.pbc
import pyscf.pbc.dft
from pyscf.scf import hf, _vhf
from pyscf import hessian

from pyscf.pbc.scf.khf import KSCF
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.multigrid import MultiGridFFTDF2

import eph
from eph.pbc import eph_fd

def gen_vnuc_deriv(cell):
    from pyscf.gto import mole
    from pyscf.pbc.grad.krhf import _make_fakemol
    from pyscf.pbc.gto.pseudo.pp import get_vlocG, _qli
    from pyscf.pbc.dft.numint import eval_ao_kpts
    from pyscf.pbc import tools

    from pyscf.pbc.grad.krhf import get_hcore
    kpts = numpy.zeros((1, 3))
    h1 = get_hcore(cell, kpts=kpts)[0]
    dtype = h1.dtype

    # Extract basic properties from the cell object
    nao = cell.nao_nr()  # Number of atomic orbitals
    aoslices = cell.aoslice_by_atom()  # Slices for each atom in AO basis
    SI = cell.get_SI()  # Structure factor [natom, grid]
    mesh = cell.mesh  # FFT mesh
    Gv = cell.Gv  # G-vectors [grid, 3]
    ngrids = len(Gv)  # Number of G-vectors
    coords = cell.get_uniform_grids()  # Real-space grid coordinates
    vlocG = get_vlocG(cell)  # Local potential in G-space [natom, grid]
    ptr = mole.PTR_ENV_START  # Starting index for environment in Mole object

    def func(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        symb = cell.atom_symbol(atm_id)
        fakemol = _make_fakemol()
        vloc_g = 1j * numpy.einsum('ga,g->ag', Gv, SI[atm_id]*vlocG[atm_id])
        nkpts = kpts.shape[0]
        nao = cell.nao_nr()
        hcore = numpy.zeros([3,nkpts,nao,nao], dtype=h1.dtype)

        kn = 0
        kpt = kpts[kn]
        ao = eval_ao_kpts(cell, coords, kpt)[0]
        rho = numpy.einsum('gi,gj->gij',ao.conj(),ao)
        for ax in range(3):
            vloc_R = tools.ifft(vloc_g[ax], mesh).real
            vloc = numpy.einsum('gij,g->ij', rho, vloc_R)
            hcore[ax,kn] += vloc
        rho = None
        aokG= tools.fftk(numpy.asarray(ao.T, order='C'),
                            mesh, numpy.exp(-1j*numpy.dot(coords, kpt))).T
        ao = None
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        if symb not in cell._pseudo:
            return hcore
        
        pp = cell._pseudo[symb]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl >0:
                hl = numpy.asarray(hl)
                fakemol._bas[0,mole.ANG_OF] = l
                fakemol._env[ptr+3] = .5*rl**2
                fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                pYlm_part = fakemol.eval_gto('GTOval', Gk)
                pYlm = numpy.empty((nl,l*2+1,ngrids))
                for k in range(nl):
                    qkl = _qli(G_rad*rl, l, k)
                    pYlm[k] = pYlm_part.T * qkl
                SPG_lmi = numpy.einsum('g,nmg->nmg', SI[atm_id].conj(), pYlm)
                SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                SPG_lmi_G = 1j * numpy.einsum('nmg, ga->anmg', SPG_lmi, Gv)
                SPG_lm_G_aoG = numpy.einsum('anmg, gp->anmp', SPG_lmi_G, aokG)
                tmp_1 = numpy.einsum('ij,ajmp->aimp', hl, SPG_lm_G_aoG)
                tmp_2 = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                vppnl = (numpy.einsum('imp,aimq->apq', SPG_lm_aoG.conj(), tmp_1) +
                            numpy.einsum('aimp,imq->apq', SPG_lm_G_aoG.conj(), tmp_2))
                vppnl *=(1./ngrids**2)
                if dtype==numpy.float64:
                    hcore[:,kn] += vppnl.real
                else:
                    hcore[:,kn] += vppnl
        # hcore[:,kn,p0:p1] -= h1[:,p0:p1]
        # hcore[:,kn,:,p0:p1] -= h1[:,p0:p1].transpose(0,2,1).conj()
        return hcore.reshape(3, nao, nao)
    
    return func

def solve_mo1(vresp, s1=None, f1=None, mo_energy=None, 
              mo_coeff=None, mo_occ=None, verbose=None,
              max_cycle=50, tol=1e-8, level_shift=0.0):
    log = logger.new_logger(verbose, None)
    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]
    nocc = orbo.shape[1]
    nvir = nmo - nocc

    def func(t1):
        t1 = t1.reshape(-1, nmo, nocc)
        dm1 = 2.0 * numpy.einsum('xpi,mp,ni->xmn', t1, mo_coeff, orbo, optimize=True)
        v1 = vresp(dm1 + dm1.transpose(0, 2, 1))
        return numpy.einsum("xmn,mp,ni->xpi", v1, mo_coeff, orbo, optimize=True)

    f1 = numpy.einsum("xmn,mp,ni->xpi", f1, mo_coeff, orbo, optimize=True)
    s1 = numpy.einsum("xmn,mp,ni->xpi", s1, mo_coeff, orbo, optimize=True)
    f1 = f1.reshape(3, nmo, nocc)
    s1 = s1.reshape(3, nmo, nocc)

    from pyscf.scf import cphf
    z1, e1 = cphf.kernel(
        func, mo_energy, mo_occ, f1, s1,
        tol=tol, max_cycle=max_cycle, 
        level_shift=level_shift,
        verbose=log
    )

    t1 = numpy.einsum('mq,xqi->xmi', mo_coeff, z1).reshape(-1, nao, nocc)
    dm1 = 2.0 * numpy.einsum('xmi,ni->xmn', t1, orbo)
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    dm1 = dm1.reshape(3, nao, nao)
    return (t1, e1), dm1

# The base for the analytic EPC calculation
class ElectronPhononCouplingBase(eph.mol.rhf.ElectronPhononCouplingBase):
    level_shift = 0.0
    conv_tol_cphf = 1e-8
    max_cycle_cphf = 50

    max_cycle = 50
    max_memory = 4000

    def __init__(self, method):
        eph.mol.rhf.ElectronPhononCouplingBase.__init__(self, method)
        self.cell = method.cell

    def gen_vnuc_deriv(self, mol=None, cell=None):
        cell = self.cell if cell is None else cell
        return gen_vnuc_deriv(cell)
    
    def gen_hcore_deriv(self, mol=None, cell=None):
        cell = self.cell if cell is None else cell
        from pyscf.pbc.grad.krhf import hcore_generator

        hcore_deriv = hcore_generator(
            self.base.to_kscf(), cell=cell,
            kpts=None
        )

        def func(ia):
            h1 = hcore_deriv(ia)
            return h1[:, 0, :, :]
        
        return func

    def gen_ovlp_deriv(self, mol=None, cell=None):
        cell = self.mol if cell is None else cell
        aoslices = cell.aoslice_by_atom()
        ipovlp = cell.pbc_intor("int1e_ipovlp")

        def func(ia):
            p0, p1 = aoslices[ia][2:]
            s1 = numpy.zeros_like(ipovlp)
            s1[:, p0:p1, :] -= ipovlp[:, p0:p1]
            s1[:, :, p0:p1] -= ipovlp[:, p0:p1].transpose(0, 2, 1)
            return s1
        return func
    
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        # print("method = ", method)
        # assert isinstance(method, scf.hf.RHF)
        assert not isinstance(method, pyscf.pbc.scf.khf.KRHF)
        assert not isinstance(method, pyscf.pbc.dft.KohnShamDFT)
        ElectronPhononCouplingBase.__init__(self, method)

    def solve_mo1(self, vresp, s1=None, f1=None, mo_energy=None,
                    mo_coeff=None, mo_occ=None, verbose=None):
        
        log = logger.new_logger(self, verbose)
        from eph.mol.rhf import solve_mo1
        return solve_mo1(
            vresp, s1=s1, f1=f1, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ, verbose=log,
            max_cycle=self.max_cycle, tol=self.conv_tol_cphf,
            level_shift=self.level_shift
        )
    
    def gen_vxc_deriv(self, mo_coeff, mo_occ):
        return None, (0.0, 0.0, 1.0, True)
    
    def gen_fock_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        scf_obj = self.base

        mol_obj = scf_obj.mol
        aoslices = mol_obj.aoslice_by_atom()

        nbas = mol_obj.nbas
        nao, nmo = mo_coeff.shape
        orbo = mo_coeff[:, mo_occ > 0]
        nocc = orbo.shape[1]
        nvir = nmo - nocc

        dm0 = scf_obj.make_rdm1(mo_coeff, mo_occ)

        tmp = self.gen_vxc_deriv(mo_coeff, mo_occ)
        vxc_deriv = tmp[0]
        omega, alpha, hyb, is_hybrid = tmp[1]
        hcore_deriv = self.gen_hcore_deriv()

        def func(ia):
            h1 = hcore_deriv(ia)

            s0, s1, p0, p1 = aoslices[ia]
            shls_slice = (s0, s1) + (0, nbas) * 3

            script_dms  = ['ji->s1kl', -dm0[:, p0:p1]] # vj1
            script_dms += ['lk->s1ij', -dm0]           # vj2

            from eph.pbc.jk import _get_jk #  shall be replaced with faster functions
            if not is_hybrid:
                vj1, vj2 = _get_jk(
                    mol_obj, 'int2e_ip1', 3, 's1',
                    script_dms=script_dms,
                    shls_slice=shls_slice
                )

                jk1 = vj1 + vj1.transpose(0, 2, 1)
                vjk = vj1
                vjk[:, p0:p1] += vj2
                vjk += vjk.transpose(0, 2, 1)

            else:
                script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1
                script_dms += ['jk->s1il', -dm0]           # vk2
            
                vj1, vj2, vk1, vk2 = _get_jk(
                    mol_obj, 'int2e_ip1', 3, 's1',
                    script_dms=script_dms,
                    shls_slice=shls_slice
                )

                jk1 = vj1 - hyb * 0.5 * vk1
                jk1 = jk1 + jk1.transpose(0, 2, 1)

                vjk = vj1 - hyb * 0.5 * vk1
                vjk[:, p0:p1] += vj2 - hyb * 0.5 * vk2
                vjk += vjk.transpose(0, 2, 1)

            f1 = h1 + vjk
            if vxc_deriv is not None:
                f1 += vxc_deriv[0, ia]
                jk1 += vxc_deriv[1, ia]

            return f1, jk1

        return func

if __name__ == '__main__':
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

    mf = scf.RHF(cell)
    mf.verbose = 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 100
    mf.kernel(dm0=None)
    dm0 = mf.make_rdm1()

    # from pyscf.pbc.grad.krhf import get_hcore, hcore_generator
    # h1 = get_hcore(cell, kpts=[numpy.zeros(3)])
    # hcore_deriv_ref = hcore_generator(
    #     mf.to_kscf(), cell=None, 
    #     kpts=numpy.zeros((1, 3))
    # )

    # hcore_deriv_sol = gen_vnuc_deriv(cell)

    # for ia in range(cell.natm):
    #     h1_sol = hcore_deriv_ref(ia)
    #     h1_ref = hcore_deriv_sol(ia)
    #     print(h1_sol.shape)
    #     print(h1_ref.shape)
    #     print(abs(h1_sol - h1_ref).max())

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol = eph_obj.kernel()

    eph_obj = eph.pbc.eph_fd.ElectronPhononCoupling(mf)
    dv_ref  = eph_obj.kernel()

    # print(dv_ref.shape)

    # err = abs(dv_sol - dv_ref).max()
    # print(err)
    for x in range(3 * cell.natm):
        err = abs(dv_sol[x] - dv_ref[x]).max()

        # if abs(dv_sol[x]).max() < 1e-6:
        #     continue
        
        print(f"\nix = {x}, error = {err:6.4e}")
        print(f"dv_sol[{x}] = ")
        numpy.savetxt(mf.stdout, dv_sol[x], fmt="% 6.4e", delimiter=", ")

        print(f"dv_ref[{x}] = ")
        numpy.savetxt(mf.stdout, dv_ref[x], fmt="% 6.4e", delimiter=", ")

        print(f"dv_sol / dv_ref = ")
        numpy.savetxt(mf.stdout, dv_sol[x] / dv_ref[x], fmt="% 6.4e", delimiter=", ")

        # assert 1 == 2