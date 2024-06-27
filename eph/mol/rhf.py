import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME

from pyscf import __config__ 
CUTOFF_FREQUENCY      = getattr(__config__, 'eph_cutoff_frequency', 80)  # 80 cm-1
KEEP_IMAG_FREQUENCY   = getattr(__config__, 'eph_keep_imaginary_frequency', False)
IMAG_CUTOFF_FREQUENCY = getattr(__config__, 'eph_imag_cutoff_frequency', 1e-4) # 1e-4 cm-1


def electron_phonon_coupling(mol, hess=None, dv_ao=None, mass=None, 
                             exclude_rot=True, exclude_trans=True, 
                             verbose=None, 
                             cutoff_frequency=CUTOFF_FREQUENCY, keep_imag_freq=KEEP_IMAG_FREQUENCY,
                             imag_cutoff_frequency=IMAG_CUTOFF_FREQUENCY):
    """
    Perform the harmonic analysis and compute the electron-phonon coupling
    from the hessian matrix and the derivative of effective potential.

    Args:
        mol : Mole
            Molecular object.
        hess : 4D array (natm, natm, 3, 3)
            Hessian matrix.
        dv_ao : 4D array (natm, 3, nao, nao)
            Derivative of effective potential.
        mass : 1D array
            Atomic masses.
        verbose : int
            Print level.

    Returns:
        res : dict
            Dictionary containing the results of the harmonic analysis
            and the electron-phonon coupling for each mode.
    """
    if mass is None:
        mass = mol.atom_mass_list()

    assert hess  is not None
    assert dv_ao is not None

    log = logger.new_logger(mol, verbose)
    natm = mol.natm
    nao = mol.nao_nr()

    assert hess.shape == (natm, natm, 3, 3)
    assert dv_ao.shape == (natm, 3, nao, nao)

    from pyscf.hessian.thermo import harmonic_analysis
    nm = harmonic_analysis(
        mol, hess, exclude_rot=exclude_rot, exclude_trans=exclude_trans, 
        mass=mass, imaginary_freq=True
        )
    
    freq = nm["freq_au"] / MP_ME ** 0.5
    # mode =  # shape (nmode, natm, 3, )
    mode = numpy.einsum("Iax,a->Iax", nm["norm_mode"], mass ** 0.5)
    nmode = len(freq)

    freq_au = freq
    freq_wn = nm["freq_wavenumber"]

    assert mode.shape == (nmode, natm, 3)
    assert freq.shape == (nmode, )

    if not keep_imag_freq:
        freq_au = freq_au.real
        freq_wn = freq_wn.real
        mask = freq_wn.real > cutoff_frequency
    else:
        mask = (freq_wn.real > cutoff_frequency) | (freq_wn.imag > imag_cutoff_frequency)

    for imode in range(nmode):
        f = "% 6.4f" % freq_wn[imode].real
        if abs(freq_wn[imode].imag) > imag_cutoff_frequency:
            f += " + " if freq_wn[imode].imag > 0 else " - "
            f += "%6.4fi" % abs(freq_wn[imode].imag)

        if not mask[imode]:
            log.info('mode %3d: %18s cm^-1 (discard)', imode, f)
        else:
            log.info('mode %3d: %18s cm^-1', imode, f)

    # sort the mask
    mask = mask[numpy.argsort(freq_au[mask].real)[::-1]]
    freq_au = freq_au[mask]
    freq_wn = freq_wn[mask]
    mode = mode[mask]

    f = 1.0 / numpy.sqrt(2 * freq_au)
    m = 1.0 / numpy.sqrt(mass * MP_ME)
    eph = numpy.einsum("axmn,Iax,I,a->Imn", dv_ao, mode, f, m, optimize=True)

    res = {
        "freq_au": freq_au, # (nmode, )
        "freq_wn": freq_wn, # (nmode, )
        "mode": mode,       # (nmode, natm, 3)
        "eph": eph          # (nmode, nao, nao)
    }

    return res

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
    nao, nmo = mo_coeff.shape

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
    
    dv = numpy.zeros((len(atmlst), 3, nao, nao))
    for i0, ia in enumerate(atmlst):
        dv[i0] = vnuc_deriv(ia) + veff_deriv(ia)

    return dv

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

class ElectronPhononCouplingBase(lib.StreamObject):
    level_shift = 0.0
    max_cycle = 50

    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.chkfile = method.chkfile

        self.mol = method.mol
        self.base = method
        self.atmlst = None

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
    
    def make_h1(self, mo_coeff, mo_occ, tmpfile=None, atmlst=None, log=None):
        raise NotImplementedError
    
    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        from pyscf.hessian.rhf import solve_mo1
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)
    
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ

        self.dump_flags(verbose=self.verbose)
        dv = kernel(
            self, mo_energy=mo_energy,
            mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=atmlst, h1ao=None, mo1=None,
        )

        return dv

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)
    
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
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-10
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    assert numpy.allclose(grad, 0.0, atol=1e-4)

    hess = mf.Hessian().kernel()

    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.verbose = 0
    dv_ao = eph_obj.kernel()

    atmlst = [0, 1]
    assert abs(dv_ao[atmlst] - eph_obj.kernel(atmlst=atmlst)).max() < 1e-6

    res = electron_phonon_coupling(
        mol, hess=hess, dv_ao=dv_ao, verbose=5, 
        exclude_rot=True, exclude_trans=True,
        keep_imag_freq=False
        )

    print(res["eph"].shape)