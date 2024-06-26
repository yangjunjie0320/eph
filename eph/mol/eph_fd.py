import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
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
                             exclude_rot=True, exclude_trans=True, verbose=None,
                             cutoff_frequency=CUTOFF_FREQUENCY,
                             keep_imag_freq=KEEP_IMAG_FREQUENCY,
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

    assert hess.shape  == (natm, natm, 3, 3)
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
        self.to_normal_mode = True

        self.max_memory = method.max_memory
        self.unit = 'au'
        self.dv_ao = None

        self.eph   = None
        self.freq  = None

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
    
    def _finalize(self, dv_ao=None, hess=None, mass=None):
        res = electron_phonon_coupling(
            self.mol, hess=hess, dv_ao=dv_ao, mass=mass
            )
        self.freq = res["freq_au"]
        self.eph  = res["eph"]
        return self

    def kernel(self):
        raise NotImplementedError

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        if atmlst is None:
            atmlst = range(self.mol.natm)

        mol = self.mol
        xyz = mol.atom_coords()
        aoslices = mol.aoslice_by_atom()

        dm0 = self.base.make_rdm1()
        nao = dm0.shape[0]
        assert dm0.shape == (nao, nao)

        grad_obj = self.base.nuc_grad_method()
        v0 = grad_obj.get_veff() + grad_obj.get_hcore() + self.base.mol.intor("int1e_ipkin")
        assert v0.shape == (3, nao, nao)

        scan_obj = self.base.as_scanner()

        dv = []
        for i0, ia in enumerate(atmlst):
            s0, s1, p0, p1 = aoslices[ia]
            for x in range(3):
                dxyz = numpy.zeros_like(xyz)
                dxyz[ia, x] = stepsize

                scan_obj(mol.set_geom_(xyz + dxyz, inplace=False, unit='B'), dm0=dm0)
                v1 = scan_obj.get_veff() + scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")

                scan_obj(mol.set_geom_(xyz - dxyz, inplace=False, unit='B'), dm0=dm0)
                v2 = scan_obj.get_veff() + scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")

                dv_ia_x = (v1 - v2) / (2 * stepsize)

                dv_ia_x[p0:p1, :] -= v0[x, p0:p1]
                dv_ia_x[:, p0:p1] -= v0[x, p0:p1].T
                dv.append(dv_ia_x)

        nao = self.mol.nao_nr()
        dv = numpy.array(dv).reshape(len(atmlst), 3, nao, nao)
        self.dv_ao = dv
        self._finalize()
        return dv

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
    eph_an = rhf.ElectronPhononCoupling(mf)
    dv_an = eph_an.kernel()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_fd = eph_fd.kernel(stepsize=stepsize).reshape(dv_an.shape)
        err = abs(dv_an - dv_fd).max()
        print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

        atmlst = [0, 1]
        assert abs(dv_fd[atmlst] - eph_fd.kernel(atmlst=atmlst, stepsize=stepsize)).max() < 1e-6

    # Test the electron-phonon coupling
    dv = dv_an
    mass = mol.atom_mass_list()
    res = electron_phonon_coupling(mol, hess=hess, dv_ao=dv, mass=mass)
