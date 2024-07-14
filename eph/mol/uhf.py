import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

import eph
import eph.mol.eph_fd, eph.mol.rhf
from eph.mol.rhf import ElectronPhononCouplingBase
from eph.mol.eph_fd import harmonic_analysis

def gen_veff_deriv(mo_occ=None, mo_coeff=None, scf_obj=None, mo1=None, h1ao=None, verbose=None):
    log = logger.new_logger(None, verbose)

    mol = scf_obj.mol
    aoslices = mol.aoslice_by_atom()
    nao, nmo = mo_coeff[0].shape
    nbas = mol.nbas
    
    ma = mo_occ[0] > 0
    mb = mo_occ[1] > 0

    orboa = mo_coeff[0][:, ma]
    orbob = mo_coeff[1][:, mb]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]

    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)

    vresp = scf_obj.gen_response(mo_coeff, mo_occ, hermi=1)

    def load(ia):
        assert mo1 is not None
        assert h1ao is not None

        t1a = None
        t1b = None

        if isinstance(mo1, str):
            assert os.path.exists(mo1), '%s not found' % mo1
            t1a = lib.chkfile.load(mo1, 'scf_mo1/0/%d' % ia)
            t1b = lib.chkfile.load(mo1, 'scf_mo1/1/%d' % ia)
            t1a = t1a.reshape(-1, nao, nocca)
            t1b = t1b.reshape(-1, nao, noccb)

        else:
            mo1a, mo1b = mo1
            t1a = mo1a[ia].reshape(-1, nao, nocca)
            t1b = mo1b[ia].reshape(-1, nao, noccb)

        assert t1a is not None
        assert t1b is not None
        t1 = (t1a, t1b)

        vj1a = None
        vk1a = None
        vj1b = None
        vk1b = None
        if isinstance(h1ao, str):
            assert os.path.exists(h1ao), '%s not found' % h1ao
            vj1a = lib.chkfile.load(h1ao, 'eph_vj1ao/0/%d' % ia)
            vk1a = lib.chkfile.load(h1ao, 'eph_vk1ao/0/%d' % ia)
            vj1b = lib.chkfile.load(h1ao, 'eph_vj1ao/1/%d' % ia)
            vk1b = lib.chkfile.load(h1ao, 'eph_vk1ao/1/%d' % ia)
        
        elif hasattr(h1ao[0][ia], 'vj1'):
            h1aoa, h1aob = h1ao

            vj1a = h1aoa[ia].vj1
            vk1a = h1aoa[ia].vk1
            vj1b = h1aob[ia].vj1
            vk1b = h1aob[ia].vk1

        if vj1a is None:
            s0, s1, p0, p1 = aoslices[ia]

            shls_slice  = (s0, s1) + (0, nbas) * 3
            script_dms  = ['ji->s2kl', -dm0a[:,p0:p1]] # vj1a
            script_dms += ['ji->s2kl', -dm0b[:,p0:p1]] # vj1b
            script_dms += ['li->s1kj', -dm0a[:,p0:p1]] # vk1a
            script_dms += ['li->s1kj', -dm0b[:,p0:p1]] # vk1b

            from pyscf.hessian.uhf import _get_jk
            tmp = _get_jk(
                mol, 'int2e_ip1', 3, 's2kl',
                script_dms=script_dms,
                shls_slice=shls_slice
            )
            
            vj1a, vj1b, vk1a, vk1b = tmp

        vj1 = vj1a + vj1b
        return t1, (vj1 - vk1a, vj1 - vk1b)

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

# TODO: implement make_h1, save some intermediate results to chkfile
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.uhf.UHF)
        ElectronPhononCouplingBase.__init__(self, method)
    
    def gen_veff_deriv(self, mo_energy=None, mo_coeff=None, mo_occ=None, 
                             scf_obj=None, mo1=None, h1ao=None, verbose=None):
        if scf_obj is None: scf_obj = self.base

        res = gen_veff_deriv(
            mo_occ=mo_occ, mo_coeff=mo_coeff, scf_obj=scf_obj,
            mo1=mo1, h1ao=h1ao, verbose=verbose
            )
        
        return res

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

    natm = mol.natm
    nao = mol.nao_nr()

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-3)
    hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    dv_sol  = eph_obj.kernel()

    # Test the finite difference against the analytic results
    eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_ref = eph_fd.kernel(stepsize=stepsize)
        err = abs(dv_sol - dv_ref).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

