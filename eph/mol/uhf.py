import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from eph.mol import eph_fd, rhf
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

def make_h1(eph_obj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = eph_obj.mol
    scf_obj = eph_obj.base

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

        from pyscf.hessian import rhf
        vj1, vj2, vk1, vk2 = rhf._get_jk(
            mol, 'int2e_ip1', 3, 's2kl',
            script_dms=script_dms,
            shls_slice=shls_slice
        )

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

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.uhf.UHF)
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

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-4)
    hess = mf.Hessian().kernel()

    from eph.mol import uhf
    eph_obj = uhf.ElectronPhononCoupling(mf)
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
