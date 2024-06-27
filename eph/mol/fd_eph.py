import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
import pyscf.eph
from pyscf.gto.mole import is_au
import pyscf.hessian
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf import hessian

from pyscf import __config__ 
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME
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

    eph = numpy.einsum(
        "axmn,Iax,I,a->Imn", dv_ao, mode,
        1.0 / numpy.sqrt(2 * freq_au),
        1.0 / numpy.sqrt(mass * MP_ME),
        optimize=True
        )

    res = {}
    for k, v in nm.items():
        if k == "freq_error":
            continue
        res[k] = v[mask]
    res["eph"] = eph

    return res

class ElectronPhononCouplingBase(lib.StreamObject):
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
    
    def kernel(self):
        raise NotImplementedError
        
class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self):
        scan = self.base.as_scanner()
        print(scan)

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
