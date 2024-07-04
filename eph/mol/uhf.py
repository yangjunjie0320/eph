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
    
def make_h1(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None, chkfile=None, atmlst=None, verbose=None):
    mol = eph_obj.mol
    scf_obj = eph_obj.base

    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    ma = mo_occ[0] > 0
    mb = mo_occ[1] > 0

    orboa = mo_coeff[0][:, ma]
    orbob = mo_coeff[1][:, mb]

    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    hcore_deriv = eph_obj.base.nuc_grad_method().hcore_generator(mol)

    nbas = mol.nbas
    aoslices = mol.aoslice_by_atom()
    h1aoa = [None] * mol.natm
    h1aob = [None] * mol.natm

    for i0, ia in enumerate(atmlst):
        s0, s1, p0, p1 = aoslices[ia]

        shls_slice  = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s2kl', -dm0a[:,p0:p1]] # vj1a
        script_dms += ['ji->s2kl', -dm0b[:,p0:p1]] # vj1b
        script_dms += ['lk->s1ij', -dm0a         ] # vj2a
        script_dms += ['lk->s1ij', -dm0b         ] # vj2b
        script_dms += ['li->s1kj', -dm0a[:,p0:p1]] # vk1a
        script_dms += ['li->s1kj', -dm0b[:,p0:p1]] # vk1b
        script_dms += ['jk->s1il', -dm0a         ] # vk2a
        script_dms += ['jk->s1il', -dm0b         ] # vk2b

        from pyscf.hessian.uhf import _get_jk
        tmp = _get_jk(
            mol, 'int2e_ip1', 3, 's2kl',
            script_dms=script_dms,
            shls_slice=shls_slice
        )
        
        vj1a, vj1b, vj2a, vj2b = tmp[:4]
        vk1a, vk1b, vk2a, vk2b = tmp[4:]
    
        vj1 = vj1a + vj1b
        vj2 = vj2a + vj2b
        vhfa = vj1 - vk1a
        vhfb = vj1 - vk1b

        vhfa[:, p0:p1] += vj2 - vk2a
        vhfb[:, p0:p1] += vj2 - vk2b

        if chkfile is None:
            h1aoa[ia] = lib.tag_array(
                vhfa + vhfa.transpose(0, 2, 1) + hcore_deriv(ia),
                vj1=vj1a, vj2=vj2a, vk1=vk1a, vk2=vk2a
            )
            
            h1aob[ia] = lib.tag_array(
                vhfb + vhfb.transpose(0, 2, 1) + hcore_deriv(ia),
                vj1=vj1b, vj2=vj2b, vk1=vk1b, vk2=vk2b
            )
        else:
            # for solve_mo1
            lib.chkfile.save(chkfile, 'scf_f1ao/0/%d' % ia, vhfa + vhfa.transpose(0,2,1) + hcore_deriv(ia))
            lib.chkfile.save(chkfile, 'scf_f1ao/1/%d' % ia, vhfb + vhfb.transpose(0,2,1) + hcore_deriv(ia))

            # for loop_vjk
            lib.chkfile.save(chkfile, 'eph_vj1ao/0/%d' % ia, vj1a)
            lib.chkfile.save(chkfile, 'eph_vj2ao/0/%d' % ia, vj2a)
            lib.chkfile.save(chkfile, 'eph_vk1ao/0/%d' % ia, vk1a)
            lib.chkfile.save(chkfile, 'eph_vk2ao/0/%d' % ia, vk2a)

            lib.chkfile.save(chkfile, 'eph_vj1ao/1/%d' % ia, vj1b)
            lib.chkfile.save(chkfile, 'eph_vj2ao/1/%d' % ia, vj2b)
            lib.chkfile.save(chkfile, 'eph_vk1ao/1/%d' % ia, vk1b)
            lib.chkfile.save(chkfile, 'eph_vk2ao/1/%d' % ia, vk2b)
    if chkfile is None:
        return (h1aoa, h1aob)
    
    else:
        return chkfile

def gen_veff_deriv(mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
    log = logger.new_logger(None, log)

    nao, nmo = mo_coeff[0].shape
    ma = mo_occ[0] > 0
    mb = mo_occ[1] > 0

    orboa = mo_coeff[0][:, ma]
    orbob = mo_coeff[1][:, mb]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]

    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)

    from pyscf.scf._response_functions import _gen_uhf_response
    vresp = _gen_uhf_response(scf_obj, mo_coeff, mo_occ, hermi=1)

    def load(ia):
        assert mo1 is not None
        if isinstance(mo1, str):
            assert os.path.exists(mo1), '%s not found' % mo1
            t1a = lib.chkfile.load(mo1, 'scf_mo1/0/%d' % ia)
            t1b = lib.chkfile.load(mo1, 'scf_mo1/1/%d' % ia)

        else:
            mo1a, mo1b = mo1
            t1a = mo1a[ia]
            t1b = mo1b[ia]

        t1 = (t1a.reshape(-1, nao, nocca), t1b.reshape(-1, nao, noccb))

        assert h1ao is not None
        if isinstance(h1ao, str):
            assert os.path.exists(h1ao), '%s not found' % h1ao
            vj1a = lib.chkfile.load(h1ao, 'eph_vj1ao/0/%d' % ia)
            vk1a = lib.chkfile.load(h1ao, 'eph_vk1ao/0/%d' % ia)
            vj1b = lib.chkfile.load(h1ao, 'eph_vj1ao/1/%d' % ia)
            vk1b = lib.chkfile.load(h1ao, 'eph_vk1ao/1/%d' % ia)
        
        else:
            h1aoa, h1aob = h1ao

            vj1a = h1aoa[ia].vj1
            vk1a = h1aoa[ia].vk1
            vj1b = h1aob[ia].vj1
            vk1b = h1aob[ia].vk1

        vj1  = vj1a + vj1b
        vjk1 = (vj1 - vk1a, vj1 - vk1b)

        return t1, vjk1

    def func(ia):
        (t1a, t1b), (vjk1a, vjk1b) = load(ia)
        dm1a = numpy.einsum('xpi,qi->xpq', t1a, orboa)
        dm1a += dm1a.transpose(0, 2, 1)

        dm1b = numpy.einsum('xpi,qi->xpq', t1b, orbob)
        dm1b += dm1b.transpose(0, 2, 1)
        dm1 = numpy.asarray((dm1a, dm1b))
        
        vinda, vindb = vresp(dm1).reshape(dm1.shape)

        vinda += vjk1a + vjk1a.transpose(0, 2, 1)
        vindb += vjk1b + vjk1b.transpose(0, 2, 1)
        return numpy.asarray((vinda, vindb)).reshape(2, 3, nao, nao)
    
    return func

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.uhf.UHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        from pyscf.hessian.uhf import solve_mo1
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
    nao = mol.nao_nr()

    atmlst = [0, 1]
    assert abs(dv_sol[atmlst] - eph_obj.kernel(atmlst=atmlst)).max() < 1e-6

    # Test the finite difference against the analytic results
    eph_fd = eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

        # dv_sol = dv_sol.reshape(-1, nao, nao)
        # dv_ref = dv_ref.reshape(-1, nao, nao)

        # print(f"{dv_sol.shape = }")
        # print(f"{dv_ref.shape = }")

        # for x in range(dv_sol.shape[0]):
        #     print(f"{x = }")
        #     numpy.savetxt(mol.stdout, dv_sol[x], fmt="% 6.4e", header="dv_sol", delimiter=", ")
        #     numpy.savetxt(mol.stdout, dv_ref[x], fmt="% 6.4e", header="dv_ref", delimiter=", ")

    # Test with the old eph code
    # res = harmonic_analysis(
    #     mol, hess=hess, dv_ao=dv_sol, mass=mol.atom_mass_list()
    # )
    # freq_sol, eph_sol = res["freq"], res["eph"]

    # eph_obj = pyscf.eph.EPH(mf)
    # eph_ref, freq_ref = eph_obj.kernel()

    # for i1, i2 in zip(numpy.argsort(freq_sol), numpy.argsort(freq_ref)):
    #     err_freq = abs(freq_sol[i1] - freq_ref[i2])
    #     assert abs(freq_sol[i1] - freq_ref[i2]) < 1e-6, "freq_sol[%d] = % 6.4e, freq_ref[%d] = % 6.4e, error = % 6.4e" % (i1, freq_sol[i1], i2, freq_ref[i2], err_freq)

    #     err_eph = abs(eph_sol[i1] - eph_ref[i2]).max()
    #     assert abs(eph_sol[i1] - eph_ref[i2]).max() < 1e-6, "eph_sol[%d], eph_ref[%d], error = % 6.4e" % (i1, i2, err_eph)


