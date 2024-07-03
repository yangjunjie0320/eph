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

    assert hess  is not None
    assert dv_ao is not None

    log = logger.new_logger(mol, verbose)
    natm = mol.natm
    nao = mol.nao_nr()

    assert hess.shape  == (natm, natm, 3, 3)
    assert dv_ao.shape == (natm, 3, nao, nao)

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
    res["freq"] = freq
 
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
    def kernel(self, atmlst=None, stepsize=1e-4, factor=1.0):
        if atmlst is None:
            atmlst = range(self.mol.natm)

        self.dump_flags()

        mol = self.mol
        xyz = mol.atom_coords()
        aoslices = mol.aoslice_by_atom()

        dm0 = self.base.make_rdm1() * factor
        nao = dm0.shape[0]
        assert dm0.shape == (nao, nao)

        grad_obj = self.base.nuc_grad_method()
        v0 = grad_obj.get_veff(dm=dm0) + grad_obj.get_hcore() + self.base.mol.intor("int1e_ipkin")
        assert v0.shape == (3, nao, nao)

        scan_obj = self.base.as_scanner()

        dv = []
        for i0, ia in enumerate(atmlst):
            s0, s1, p0, p1 = aoslices[ia]
            for x in range(3):
                dxyz = numpy.zeros_like(xyz)
                dxyz[ia, x] = stepsize

                scan_obj(mol.set_geom_(xyz + dxyz, inplace=False, unit='B'), dm0=dm0)
                dm1 = scan_obj.make_rdm1() * factor
                v1  = scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")
                # v1 += scan_obj.get_veff(dm=dm1)

                scan_obj(mol.set_geom_(xyz - dxyz, inplace=False, unit='B'), dm0=dm0)
                dm2 = scan_obj.make_rdm1() * factor
                v2  = scan_obj.get_hcore() - scan_obj.mol.intor_symmetric("int1e_kin")
                # v2 += scan_obj.get_veff(dm=dm2)

                dv_ia_x = (v1 - v2) / (2 * stepsize)

                # dv_ia_x[p0:p1, :] -= v0[x, p0:p1]
                # dv_ia_x[:, p0:p1] -= v0[x, p0:p1].T
                dv.append(dv_ia_x)

        nao = self.mol.nao_nr()
        dv = numpy.array(dv).reshape(len(atmlst), 3, nao, nao)
        self.dv_ao = dv
        return dv
    
def dveff_to_dvnuc(dv, mf_obj=None):
    nocc = numpy.count_nonzero(mf_obj.mo_occ)
    orbo = mf_obj.mo_coeff[:,mf_obj.mo_occ>0]
    orbv = mf_obj.mo_coeff[:,mf_obj.mo_occ==0]
    e_o = mf_obj.mo_energy[mf_obj.mo_occ>0]
    e_v = mf_obj.mo_energy[mf_obj.mo_occ==0]

    dv_vo = numpy.einsum('mn,ma,ni->ai', dv, orbv, orbo)
    e_vo = e_v[:, None] - e_o
    ddm = orbv @ (dv_vo / e_vo) @ orbo.T
    ddm = - (ddm + ddm.T) * 2

    vresp = mf_obj.gen_response(hermi=0)
    dv0 = dv - vresp(ddm)
    return dv0

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       0.0000000000     0.0000000000     0.1146878262
    H      -0.7540663886    -0.0000000000    -0.4587203947
    H       0.7540663886    -0.0000000000    -0.4587203947
    '''
    mol.basis = 'sto3g' # '631g*'
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

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-4)
    hess = mf.Hessian().kernel()

    from eph.mol import rhf
    eph_an = rhf.ElectronPhononCoupling(mf)
    dv_an = eph_an.kernel()
    nao = mol.nao_nr()

    # Test the finite difference against the analytic results
    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0

    from pyscf.scf._response_functions import _gen_rhf_response
    vresp = _gen_rhf_response(mf, mf.mo_coeff, mf.mo_occ, hermi=1)

    for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
        dv_fd = eph_fd.kernel(stepsize=stepsize).reshape(dv_an.shape)
        err = abs(dv_an - dv_fd).max()
        # print("\nstepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

        # atmlst = [0, 1]
        # assert abs(dv_fd[atmlst] - eph_fd.kernel(atmlst=atmlst, stepsize=stepsize)).max() < 1e-6

        dveff_fd = dv_fd.reshape(-1, nao, nao)
        dvnuc_fd = eph_fd.kernel(stepsize=stepsize, factor=0.0).reshape(dveff_fd.shape)

        nn = dveff_fd.shape[0]
        for xx in range(nn):
            dvnuc_an_xx = dveff_to_dvnuc(dveff_fd[xx], mf_obj=mf)
            err = abs(dvnuc_an_xx - dvnuc_fd[xx]).max()

            print("\nxx = %2d, stepsize = % 6.4e, error = % 6.4e" % (xx, stepsize, err))
            numpy.savetxt(mol.stdout, dvnuc_an_xx, fmt='% 12.8f', delimiter=',', header='dvnuc_an_xx')
            numpy.savetxt(mol.stdout, dvnuc_fd[xx], fmt='% 12.8f', delimiter=',', header='dvnuc_fd')

    assert 1 == 2
            


    # Test the electron-phonon coupling
    dv = dv_an
    mass = mol.atom_mass_list()
    res = harmonic_analysis(mol, hess=hess, dv_ao=dv, mass=mass)
