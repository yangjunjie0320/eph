import os, numpy, scipy, tempfile

import pyscf
from pyscf import lib, scf
from pyscf.gto.mole import is_au
from pyscf.lib import logger

from pyscf import __config__
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME
CUTOFF_FREQUENCY      = getattr(__config__, 'eph_cutoff_frequency', 80)  # 80 cm-1
KEEP_IMAG_FREQUENCY   = getattr(__config__, 'eph_keep_imaginary_frequency', False)
IMAG_CUTOFF_FREQUENCY = getattr(__config__, 'eph_imag_cutoff_frequency', 1e-4) # 1e-4 cm-1

def harmonic_analysis(mol, hess=None, dv_ao=None, mass=None,
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

    log = logger.new_logger(mol, verbose)
    natm = mol.natm
    nao = mol.nao_nr()
    dv_ao = numpy.asarray(dv_ao).reshape(-1, natm, 3, nao, nao)
    spin = dv_ao.shape[0]

    assert hess.shape  == (natm, natm, 3, 3)
    assert dv_ao.shape == (spin, natm, 3, nao, nao)

    from pyscf.hessian import thermo
    nm = thermo.harmonic_analysis(
        mol, hess, exclude_rot=exclude_rot,
        exclude_trans=exclude_trans,
        mass=mass, imaginary_freq=True
        )
    
    freq = nm["freq_au"] / MP_ME ** 0.5
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

    freq_au = freq_au[mask]
    freq_wn = freq_wn[mask]
    mode = mode[mask]
    nmode = len(freq_au)

    eph = numpy.einsum(
        "saxmn,Iax,I,a->sImn", dv_ao, mode,
        1.0 / numpy.sqrt(2 * freq_au),
        1.0 / numpy.sqrt(mass * MP_ME),
        optimize=True
        )

    eph = eph.reshape(spin, -1, nao, nao)

    if spin == 1:
        eph = eph[0]

    res = {}
    for k, v in nm.items():
        if k == "freq_error":
            continue
        res[k] = v[mask]

    res["eph"] = eph
    res["freq"] = freq_au
 
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
        self.dv_ao = None

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
    
    def _finalize(self, dv_ao):
        assert dv_ao is not None
        if not isinstance(dv_ao, numpy.ndarray):
            spin = dv_ao[0].shape[0]
            nao = dv_ao[0].shape[-1]
            dv_ao = numpy.asarray(dv_ao)
            dv_ao = dv_ao.reshape(-1, 3, spin, nao, nao)

        natm, _, spin, nao, _ = dv_ao.shape
        assert dv_ao.shape == (natm, 3, spin, nao, nao)

        if spin == 1:
            dv_ao = dv_ao.reshape(-1, nao, nao)
        
        elif spin == 2:
            dv_ao = dv_ao.reshape(-1, 2, nao, nao)
            dv_ao = numpy.asarray((dv_ao[:, 0], dv_ao[:, 1]))
            assert dv_ao.shape == (spin, natm * 3, nao, nao)

        else:
            raise RuntimeError("spin = %d is not supported" % spin)

        return dv_ao

    def kernel(self):
        raise NotImplementedError
    
def _fd(scan_obj=None, ix=None, atmlst=None, stepsize=1e-4, v0=None, dm0=None, xyz=None):
    ia = atmlst[ix // 3]
    x = ix % 3
    nao = scan_obj.mol.nao_nr()
    p0, p1 = scan_obj.mol.aoslice_by_atom()[ia][2:]

    dm0 = dm0.reshape(-1, nao, nao)
    spin = dm0.shape[0]
    assert v0.shape == (spin, 3, nao, nao)

    dxyz = numpy.zeros_like(xyz)
    dxyz[ia, x] = stepsize

    m1 = scan_obj.mol.set_geom_(xyz + dxyz, unit="Bohr", inplace=False)
    scan_obj(m1, dm0=dm0[0] if spin == 1 else dm0)
    dm1 = scan_obj.make_rdm1()
    v1  = scan_obj.get_veff(dm=dm1).reshape(spin, nao, nao)
    v1 += scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")
    
    m2 = scan_obj.mol.set_geom_(xyz - dxyz, unit="Bohr", inplace=False)
    scan_obj(m2, dm0=dm0[0] if spin == 1 else dm0)
    dm2 = scan_obj.make_rdm1()
    v2  = scan_obj.get_veff(dm=dm2).reshape(spin, nao, nao)
    v2 += scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")

    assert v1.shape == v2.shape == (spin, nao, nao)

    dv_ia_x = (v1 - v2) / (2 * stepsize)
    dv_ia_x[:, p0:p1, :] -= v0[:, x, p0:p1, :]
    dv_ia_x[:, :, p0:p1] -= v0[:, x, p0:p1, :].transpose(0, 2, 1)
    return dv_ia_x.reshape(spin, nao, nao)

class ElectronPhononCoupling(ElectronPhononCouplingBase):
    def kernel(self, atmlst=None, stepsize=1e-4):
        mol = self.mol
        nao = mol.nao_nr()
        xyz = mol.atom_coords(unit="Bohr")

        if atmlst is None:
            atmlst = range(mol.natm)
        
        natm = len(atmlst)
        self.dump_flags()

        scf_obj = self.base
        scan_obj = scf_obj.as_scanner()
        grad_obj = scf_obj.nuc_grad_method()

        dm0 = scf_obj.make_rdm1()
        dm0 = dm0.reshape(-1, nao, nao)
        spin = dm0.shape[0]

        v0  = grad_obj.get_veff(dm=dm0[0] if spin == 1 else dm0)
        v0  = v0.reshape(spin, 3, nao, nao)
        v0 += grad_obj.get_hcore()
        v0 += mol.intor("int1e_ipkin")
    
        dv_ao = []
        for ix in range(3 * natm):
            dv_ia_x = _fd(
                scan_obj=scan_obj, xyz=xyz,
                ix=ix, atmlst=atmlst, 
                stepsize=stepsize,
                v0=v0, dm0=dm0, 
            )
            dv_ao.append(dv_ia_x)

        self.dv_ao = self._finalize(dv_ao)
        return self.dv_ao

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

    # Test the finite difference against the analytic results
    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.verbose = 0
    dv_fd = eph_obj.kernel(stepsize=1e-4)

    res = harmonic_analysis(mol, hess=mf.Hessian().kernel(), dv_ao=dv_fd)
