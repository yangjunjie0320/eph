import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from eph.mol import eph_fd
from eph.mol.eph_fd import harmonic_analysis

def kernel(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None,
           h1ao=None, mo1=None, atmlst=None,
           max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol
    scf_obj = eph_obj.base

    if mo_energy is None: mo_energy = scf_obj.mo_energy
    if mo_occ    is None: mo_occ    = scf_obj.mo_occ
    if mo_coeff  is None: mo_coeff  = scf_obj.mo_coeff
    if atmlst is None:    atmlst    = range(mol_obj.natm)
    nao, nmo = mo_coeff.shape[-2:]

    if h1ao is None:
        h1ao = eph_obj.make_h1(mo_coeff, mo_occ, eph_obj.chkfile, atmlst, log)
        t1 = log.timer_debug1('making H1', *t0)

    if mo1 is None:
        mo1, mo_e1 = eph_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol_obj)
    veff_deriv = eph_obj.gen_veff_deriv(
        mo_occ, mo_coeff, scf_obj=scf_obj,
        mo1=mo1, h1ao=h1ao, log=log
        )
    
    dv = [] # numpy.zeros((len(atmlst), 3, nao, nao))
    for i0, ia in enumerate(atmlst):
        vnuc = vnuc_deriv(ia)
        veff = veff_deriv(ia)
        dv.append(vnuc + veff)

    return dv

def make_h1(eph_obj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = eph_obj.mol
    scf_obj = eph_obj.base
    
    if atmlst is None:
        atmlst = range(mol.natm)

    mask = mo_occ > 0
    orbo = mo_coeff[:, mask]
    dm0 = numpy.dot(orbo, orbo.T) * 2.0

    nbas = mol.nbas
    aoslices = mol.aoslice_by_atom()
    h1ao = [None] * mol.natm

    hcore_deriv = scf_obj.nuc_grad_method().hcore_generator(mol)
    for i0, ia in enumerate(atmlst):
        s0, s1, p0, p1 = aoslices[ia]

        shls_slice  = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s2kl', -dm0[:, p0:p1]] # vj1
        script_dms += ['lk->s1ij', -dm0]           # vj2
        script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1
        script_dms += ['jk->s1il', -dm0]           # vk2

        from pyscf.hessian.rhf import _get_jk
        tmp = _get_jk(
            mol, 'int2e_ip1', 3, 's2kl',
            script_dms=script_dms,
            shls_slice=shls_slice
        )

        vj1, vj2, vk1, vk2 = tmp
        vhf = vj1 - vk1 * 0.5
        vhf[:, p0:p1] += vj2 - vk2 * 0.5

        if chkfile is None:
            h1ao[ia] = lib.tag_array(
                vhf + vhf.transpose(0, 2, 1) + hcore_deriv(ia),
                vj1=vj1, vj2=vj2, vk1=vk1, vk2=vk2
            )

        else:
            # for solve_mo1
            lib.chkfile.save(chkfile, 'scf_f1ao/%d' % ia, vhf + vhf.transpose(0,2,1) + hcore_deriv(ia))

            # for loop_vjk
            lib.chkfile.save(chkfile, 'eph_vj1ao/%d' % ia, vj1)
            lib.chkfile.save(chkfile, 'eph_vj2ao/%d' % ia, vj2)
            lib.chkfile.save(chkfile, 'eph_vk1ao/%d' % ia, vk1)
            lib.chkfile.save(chkfile, 'eph_vk2ao/%d' % ia, vk2)

    if chkfile is None:
        return h1ao
    
    else:
        return chkfile

def gen_vnuc_deriv(mol):
    def func(ia):
        with mol.with_rinv_at_nucleus(ia):
            vrinv  =  mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(ia)
        return vrinv + vrinv.transpose(0, 2, 1)
    return func

def gen_veff_deriv(mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
    log = logger.new_logger(None, log)
    nao = mo_coeff.shape[0]

    nao, nmo = mo_coeff.shape
    mask = mo_occ > 0
    orbo = mo_coeff[:, mask]
    nocc = orbo.shape[1]

    from pyscf.scf._response_functions import _gen_rhf_response
    vresp = _gen_rhf_response(scf_obj, mo_coeff, mo_occ, hermi=1)

    def load(ia):
        assert mo1 is not None
        if isinstance(mo1, str):
            assert os.path.exists(mo1), '%s not found' % mo1
            t1 = lib.chkfile.load(mo1, 'scf_mo1/%d' % ia)
            t1 = t1.reshape(-1, nao, nocc)

        else:
            t1 = mo1[ia].reshape(-1, nao, nocc)

        assert h1ao is not None
        if isinstance(h1ao, str):
            assert os.path.exists(h1ao), '%s not found' % h1ao
            vj1 = lib.chkfile.load(h1ao, 'eph_vj1ao/%d' % ia)
            vk1 = lib.chkfile.load(h1ao, 'eph_vk1ao/%d' % ia)
        
        else:
            vj1 = h1ao[ia].vj1
            vk1 = h1ao[ia].vk1

        assert t1 is not None
        assert vj1 is not None
        assert vk1 is not None

        return t1, vj1 - vk1 * 0.5

    def func(ia):
        t1, vjk1 = load(ia)
        dm1 = 2.0 * numpy.einsum('xpi,qi->xpq', t1, orbo)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        return vresp(dm1) + vjk1 + vjk1.transpose(0, 2, 1)
    
    return func

# The base for the analytic EPC calculation
class ElectronPhononCouplingBase(eph_fd.ElectronPhononCouplingBase):
    level_shift = 0.0
    max_cycle = 50

    def gen_vnuc_deriv(self, mol=None):
        if mol is None: mol = self.mol
        return gen_vnuc_deriv(mol)

    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
        raise NotImplementedError

    def make_h1(self, mo_coeff, mo_occ, tmpfile=None, atmlst=None, log=None):
        raise NotImplementedError

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ

        self.dump_flags()
        dv = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, h1ao=None, mo1=None,
        )

        self.dv_ao = dv
        return dv

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        from pyscf.hessian.rhf import solve_mo1
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    
    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
        return gen_veff_deriv(mo_occ, mo_coeff, scf_obj=scf_obj, mo1=mo1, h1ao=h1ao, log=log)
    
    make_h1 = make_h1
    
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

    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-4)
    hess = mf.Hessian().kernel()

    from eph.mol import rhf
    eph_obj = rhf.ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()

    atmlst = [0, 1]
    assert abs(dv_sol[atmlst] - eph_obj.kernel(atmlst=atmlst)).max() < 1e-6

    # Test the finite difference against the analytic results
    eph_fd = eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

    # Test with the old eph code
    res = harmonic_analysis(
        mol, hess=hess, dv_ao=dv_sol, mass=mol.atom_mass_list()
    )
    freq_sol, eph_sol = res["freq"], res["eph"]

    eph_obj = pyscf.eph.EPH(mf)
    eph_ref, freq_ref = eph_obj.kernel()

    for i1, i2 in zip(numpy.argsort(freq_sol), numpy.argsort(freq_ref)):
        err_freq = abs(freq_sol[i1] - freq_ref[i2])
        assert abs(freq_sol[i1] - freq_ref[i2]) < 1e-6, "freq_sol[%d] = % 6.4e, freq_ref[%d] = % 6.4e, error = % 6.4e" % (i1, freq_sol[i1], i2, freq_ref[i2], err_freq)

        err_eph = abs(eph_sol[i1] - eph_ref[i2]).max()
        assert abs(eph_sol[i1] - eph_ref[i2]).max() < 1e-6, "eph_sol[%d], eph_ref[%d], error = % 6.4e" % (i1, i2, err_eph)
