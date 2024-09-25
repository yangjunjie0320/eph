import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf, dft
from pyscf.scf import cphf
from pyscf.lib import logger

from pyscf import hessian
from pyscf.hessian.rhf import _get_jk

import eph
import eph.mol.eph_fd
from eph.mol.eph_fd import harmonic_analysis

def kernel(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None,
           chkfile=None, atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol

    scf_obj = eph_obj.base
    if mo_energy is None: mo_energy = scf_obj.mo_energy
    if mo_occ    is None: mo_occ    = scf_obj.mo_occ
    if mo_coeff  is None: mo_coeff  = scf_obj.mo_coeff
    if atmlst is None:    atmlst    = range(mol_obj.natm)
    
    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol=mol_obj)
    ovlp_deriv = eph_obj.gen_ovlp_deriv(mol=mol_obj)
    fock_deriv = eph_obj.gen_fock_deriv(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    vresp = scf_obj.gen_response(mo_coeff, mo_occ, hermi=1)
    
    dv_ao = [] # numpy.zeros((len(atmlst), 3, nao, nao))
    for i0, ia in enumerate(atmlst):
        v1 = vnuc_deriv(ia)
        v1 += v1.transpose(0, 2, 1)

        s1 = ovlp_deriv(ia)
        f1, jk1 = fock_deriv(ia)
        v1 += jk1

        (t1, e1), dm1 = eph_obj.solve_mo1(
            vresp, s1=s1, f1=f1,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ, verbose=log
        )

        v1 += vresp(dm1)
        dv_ao.append(v1)

    return dv_ao

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
class ElectronPhononCouplingBase(eph.mol.eph_fd.ElectronPhononCouplingBase):
    level_shift = 0.0
    conv_tol_cphf = 1e-8
    max_cycle_cphf = 50

    max_cycle = 50
    max_memory = 4000

    def gen_vnuc_deriv(self, mol=None):
        mol = self.mol if mol is None else mol

        def func(ia):
            with mol.with_rinv_at_nucleus(ia):
                vrinv  =  mol.intor('int1e_iprinv', comp=3)
                vrinv *= -mol.atom_charge(ia)
            return vrinv # + vrinv.transpose(0, 2, 1)
        
        return func
    
    def gen_kine_deriv(self, mol=None):
        mol = self.mol if mol is None else mol
        aoslices = mol.aoslice_by_atom()
        ipkin = mol.intor("int1e_ipkin")

        def func(ia):
            p0, p1 = aoslices[ia][2:]
            s1 = numpy.zeros_like(ipkin)
            s1[:, p0:p1, :] -= ipkin[:, p0:p1]
            s1[:, :, p0:p1] -= ipkin[:, p0:p1].transpose(0, 2, 1)
            return s1
        return func

    def gen_ovlp_deriv(self, mol=None):
        mol = self.mol if mol is None else mol
        aoslices = mol.aoslice_by_atom()
        ipovlp = mol.intor("int1e_ipovlp")

        def func(ia):
            p0, p1 = aoslices[ia][2:]
            s1 = numpy.zeros_like(ipovlp)
            s1[:, p0:p1, :] -= ipovlp[:, p0:p1]
            s1[:, :, p0:p1] -= ipovlp[:, p0:p1].transpose(0, 2, 1)
            return s1
        return func
    
    def _hcore_deriv(self, mol=None):
        mol = self.mol if mol is None else mol
        grad_obj = self.base.nuc_grad_method()
        return grad_obj.get_hcore(mol=mol)
    
    def gen_hcore_deriv(self, mol=None):
        mol = self.mol if mol is None else mol
        ao_slices = mol.aoslice_by_atom()
        h1 = self._hcore_deriv(mol=mol)
        vnuc_deriv = self.gen_vnuc_deriv(mol=mol)

        def func(ia):
            s0, s1, p0, p1 = ao_slices[ia]
            dv = vnuc_deriv(ia)
            dv[:, p0:p1] += h1[:, p0:p1]
            return dv + dv.transpose(0, 2, 1).conj()
        
        return func
        
    def gen_fock_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        raise NotImplementedError
    
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None:  mo_coeff = self.base.mo_coeff
        if mo_occ is None:    mo_occ = self.base.mo_occ

        self.dump_flags()
        dv_ao = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, chkfile=self.chkfile,
        )

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

# TODO: implement make_h1, save some intermediate results to chkfile
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def solve_mo1(self, vresp, s1=None, f1=None, mo_energy=None,
                    mo_coeff=None, mo_occ=None, verbose=None):
        
        log = logger.new_logger(self, verbose)

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

        hcore_deriv = self.gen_hcore_deriv(mol=mol_obj)

        def func(ia):
            h1 = hcore_deriv(ia)

            s0, s1, p0, p1 = aoslices[ia]
            shls_slice = (s0, s1) + (0, nbas) * 3

            script_dms  = ['ji->s2kl', -dm0[:, p0:p1]] # vj1
            script_dms += ['lk->s1ij', -dm0]           # vj2

            if not is_hybrid:
                vj1, vj2 = _get_jk(
                    mol_obj, 'int2e_ip1', 3, 's2kl',
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
                    mol_obj, 'int2e_ip1', 3, 's2kl',
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
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       0.0000000000     0.0000000000     0.1146878262
    H      -0.7540663886    -0.0000000000    -0.4587203947
    H       0.7540663886    -0.0000000000    -0.4587203947
    '''
    mol.basis = 'sto3g' # 631g*'
    mol.verbose = 0
    mol.symmetry = False
    mol.cart = True
    mol.unit = "AA"
    mol.build()

    natm = mol.natm
    nao = mol.nao_nr()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-3)
    # hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))
