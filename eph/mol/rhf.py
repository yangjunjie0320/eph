import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME

from pyscf import __config__ 
CUTOFF_FREQUENCY      = getattr(__config__, 'eph_cutoff_frequency', 80)  # 80 cm-1
KEEP_IMAG_FREQUENCY   = getattr(__config__, 'eph_keep_imaginary_frequency', False)
IMAG_CUTOFF_FREQUENCY = getattr(__config__, 'eph_imag_cutoff_frequency', 1e-4)

def kernel(eph_obj, mo_energy=None, mo_coeff=None, mo_occ=None,
           h1ao=None, mo1=None, max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol
    scf_obj = eph_obj.base

    if mo_energy is None: mo_energy = scf_obj.mo_energy
    if mo_occ    is None: mo_occ    = scf_obj.mo_occ
    if mo_coeff  is None: mo_coeff  = scf_obj.mo_coeff
    nao, nmo = mo_coeff.shape
          
    # Check if the mo1 is provided
    if h1ao is None or mo1 is None:
        mo1_to_save = eph_obj.chkfile
        if mo1_to_save is not None:
            log.debug('mo1 will be saved in %s', mo1_to_save)

        h1ao, mo1, mo_e1 = eph_obj.solve_mo1(
            mo_energy, mo_coeff, mo_occ, log=log,
            tmpfile=mo1_to_save
        )
        t1 = log.timer('solving CP-SCF equations', *t0)

    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol_obj)
    veff_deriv = eph_obj.gen_veff_deriv(
        mo_occ, mo_coeff, scf_obj=scf_obj,
        mo1=mo1, h1ao=h1ao, log=log
        )
    
    natm = mol_obj.natm
    dv = numpy.zeros((natm, 3, nao, nao))
    for ia in range(natm):
        dv[ia] = vnuc_deriv(ia) + veff_deriv(ia)
    return dv.reshape(-1, nao, nao)

def make_h1(mo_energy=None, mo_coeff=None, mo_occ=None, tmpfile=None, scf_obj=None, log=None):
    mol = scf_obj.mol
    nbas = mol.nbas

    mask = mo_occ > 0
    orbo = mo_coeff[:, mask]
    dm0 = numpy.dot(orbo, orbo.T) * 2.0
    hcore_deriv = scf_obj.nuc_grad_method().hcore_generator(mol)

    h1 = []
    if tmpfile is not None:
        h1 = tmpfile

    aoslices = mol.aoslice_by_atom()
    for ia, (s0, s1, p0, p1) in enumerate(aoslices):
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


        if tmpfile is not None:
            # for solve_mo1
            lib.chkfile.save(tmpfile, 'scf_f1ao/%d' % ia, vhf + vhf.transpose(0,2,1) + hcore_deriv(ia))

            # for loop_vjk
            lib.chkfile.save(tmpfile, 'eph_vj1ao/%d' % ia, vj1)
            lib.chkfile.save(tmpfile, 'eph_vj2ao/%d' % ia, vj2)
            lib.chkfile.save(tmpfile, 'eph_vk1ao/%d' % ia, vk1)
            lib.chkfile.save(tmpfile, 'eph_vk2ao/%d' % ia, vk2)

        else:
            vhf = lib.tag_array(
                vhf + vhf.transpose(0,2,1) + hcore_deriv(ia),
                vj1=vj1, vj2=vj2, vk1=vk1, vk2=vk2
            )
            h1.append(vhf)

    return h1

def gen_vnuc_deriv(mol):
    def func(ia):
        with mol.with_rinv_at_nucleus(ia):
            vrinv  =  mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(ia)
        return vrinv + vrinv.transpose(0, 2, 1)
    return func

def gen_veff_deriv(mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
    log = logger.new_logger(None, log)
    mol = scf_obj.mol
    nao = mo_coeff.shape[0]
    natm = mol.natm

    nao, nmo = mo_coeff.shape
    mask = mo_occ > 0
    orbo = mo_coeff[:, mask]
    nocc = orbo.shape[1]

    from pyscf.scf._response_functions import _gen_rhf_response
    vresp = _gen_rhf_response(mf, mo_coeff, mo_occ, hermi=1)

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

        return t1, vj1 - vk1 * 0.5

    def func(ia):
        t1, vjk1 = load(ia)
        dm1 = 2.0 * numpy.einsum('xpi,qi->xpq', t1, orbo)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        return vresp(dm1) + vjk1 + vjk1.transpose(0, 2, 1)
    
    return func

def electron_phonon_coupling(mol, hess=None, dv_ao=None, mass=None, verbose=None):
    assert hess  is not None
    assert dv_ao is not None

    log = logger.new_logger(mol, verbose)
    natm = mol.natm
    nao = mol.nao_nr()

    assert hess.shape == (natm, natm, 3, 3)
    assert dv_ao.shape == (natm, 3, nao, nao)

    from pyscf.hessian.thermo import harmonic_analysis
    nm = harmonic_analysis(
        mol, hess, exclude_rot=False, exclude_trans=False, 
        mass=mass, imaginary_freq=True
        )

    print(nm)

class ElectronPhononCouplingBase(lib.StreamObject):
    level_shift = 0.0
    max_cycle = 50

    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.chkfile = method.chkfile

        self.mol = method.mol
        self.base = method

        self.max_memory = method.max_memory
        self.unit = 'au'
        self.dveff  = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')

        base = self.base
        if hasattr(base, 'converged') and not base.converged:
            log.warn('Ground state %s not converged', base.__class__.__name__)
            
        log.info('******** %s for %s ********', self.__class__, base.__class__)
        
        if not is_au(self.unit):
            raise NotImplementedError('unit Eh/Ang is not supported')
        else:
            log.info('unit = Eh/Bohr')

        log.info(
            'max_memory %d MB (current use %d MB)',
            self.max_memory, lib.current_memory()[0]
            )
        return self
    
    def gen_vnuc_deriv(self, mol=None):
        if mol is None: mol = self.mol
        return gen_vnuc_deriv(mol)
    
    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
        raise NotImplementedError
    
    def solve_mo1(self, *args, **kwargs):
        raise NotImplementedError
    
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ

        dv = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            h1ao=None, mo1=None
        )

        return dv

class RHF(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, tmpfile=None, log=None):
        h1ao = make_h1(mo_energy, mo_coeff, mo_occ, tmpfile=tmpfile, scf_obj=self.base, log=log)
        mo1, mo_e1 = hessian.rhf.solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao, fx=None)
        return h1ao, mo1, mo_e1
    
    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, mo1=None, h1ao=None, log=None):
        return gen_veff_deriv(mo_occ, mo_coeff, scf_obj=scf_obj, mo1=mo1, h1ao=h1ao, log=log)
    
if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       1.4877130648    -0.0141244699     0.1077221832
    H       1.3603537501     0.8074302059    -0.4276645268
    H       1.2382957299    -0.6943961257    -0.5650025559
    O      -1.4874109631    -0.0141053847    -0.1078225865
    H      -1.3614968377     0.8073033485     0.4281140554
    H      -1.2395694562    -0.6944985917     0.5653562536
    '''
    mol.basis = 'sto3g'
    mol.verbose = 5
    mol.build()

    natm = mol.natm

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-4)

    eph_obj = RHF(mf)
    eph_obj.verbose = 5
    eph_obj.chkfile = None
    eph_sol = eph_obj.kernel()
    # chkfile = eph_obj.chkfile
    # assert 1 == 2

    # from pyscf.eph.rhf import EPH
    # from pyscf.eph.rhf import get_eph
    # def func(*args):
    #     print("hack _freq_mass_weighted_vec")
    #     print(args)
    #     return numpy.eye(natm * 3)
    # pyscf.eph.rhf._freq_mass_weighted_vec = func
    # eph_obj = EPH(mf)
    # eph_ref = get_eph(eph_obj, chkfile, omega=None, vec=None, mo_rep=False)

    # err = numpy.linalg.norm(eph_sol - eph_ref)
    # print('eph error %6.4e' % err)

    hess_obj = hessian.RHF(mf)
    hess_obj.verbose = 5

    hess = hess_obj.kernel()

    # from pyscf.hessian.thermo import harmonic_analysis
    # mol.verbose = 5
    # res = harmonic_analysis(mol, hess)

    # print(res)
