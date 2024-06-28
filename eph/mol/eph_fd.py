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
    def kernel(self, atmlst=None, stepsize=1e-4):
        if atmlst is None:
            atmlst = range(self.mol.natm)

        mol = self.mol
        xyz = mol.atom_coords()
        aoslices = mol.aoslice_by_atom()

        scf_obj = self.base

        dm0 = self.base.make_rdm1()
        nao = dm0.shape[0]
        assert dm0.shape == (nao, nao)

        grad_obj = self.base.nuc_grad_method()
        v0 = grad_obj.get_veff() + grad_obj.get_hcore() + self.base.mol.intor("int1e_ipkin")
        assert v0.shape == (3, nao, nao)

        scanner = self.base.as_scanner()
        scanner.verbose = 0

        dv = []
        for i0, ia in enumerate(atmlst):
            s0, s1, p0, p1 = aoslices[ia]
            for x in range(3):
                dxyz = numpy.zeros_like(xyz)
                dxyz[ia, x] = stepsize

                mol1 = mol.set_geom_(xyz + dxyz, inplace=False, unit='B')
                mol2 = mol.set_geom_(xyz - dxyz, inplace=False, unit='B')

                print("Computing %d-th atom, %d-th component" % (ia, x))

                mf1 = scf.RHF(mol1)
                mf1.__dict__.update(scf_obj.__dict__)
                mf1.kernel(dm0)

                # v1 = mf1.get_veff() + mf1.get_hcore() - mf1.mol.intor("int1e_kin")

                mf2 = scf.RHF(mol2)
                mf2.__dict__.update(scf_obj.__dict__)
                mf2.kernel(dm0)

                # v2 = mf2.get_veff() + mf2.get_hcore() - mf2.mol.intor("int1e_kin")

                # dv_ia_x = (v1 - v2) / (2 * stepsize)

                # dv_ia_x[p0:p1, :] -= v0[x, p0:p1]
                # dv_ia_x[:, p0:p1] -= v0[x, p0:p1].T

                # dv.append(dv_ia_x)


        nao = self.mol.nao_nr()
        dv = numpy.array(dv).reshape(len(atmlst), 3, nao, nao)
        return dv
    
def get_vmat(mf, mfset, disp):
    vmat=[]
    mygrad = mf.nuc_grad_method()
    ve = mygrad.get_veff() + mygrad.get_hcore() + mf.mol.intor("int1e_ipkin")
    RESTRICTED = (ve.ndim==3)
    aoslice = mf.mol.aoslice_by_atom()
    for ki, (mf1, mf2) in enumerate(mfset):
        atmid, axis = numpy.divmod(ki, 3)
        p0, p1 = aoslice[atmid][2:]
        vfull1 = mf1.get_veff() + mf1.get_hcore() - mf1.mol.intor_symmetric('int1e_kin')  # <u+|V+|v+>
        vfull2 = mf2.get_veff() + mf2.get_hcore() - mf2.mol.intor_symmetric('int1e_kin')  # <u-|V-|v->
        vfull = (vfull1 - vfull2)/disp  # (<p+|V+|q+>-<p-|V-|q->)/dR

        print("ki = %d, atmid = %d, axis = %d, p0 = %d, p1 = %d" % (ki, atmid, axis, p0, p1))
        numpy.savetxt(mf.mol.stdout, vfull, fmt="% 6.4e", delimiter=", ", header="vfull1")

        if RESTRICTED:
            vfull[p0:p1] -= ve[axis,p0:p1]
            vfull[:,p0:p1] -= ve[axis,p0:p1].T
        else:
            vfull[:,p0:p1] -= ve[:,axis,p0:p1]
            vfull[:,:,p0:p1] -= ve[:,axis,p0:p1].transpose(0,2,1)
        vmat.append(vfull)

    return numpy.asarray(vmat)

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O       0.0000000000     0.0000000000     0.1146878262
    H      -0.7540663886    -0.0000000000    -0.4587203947
    H       0.7540663886    -0.0000000000    -0.4587203947
    '''
    mol.basis = 'sto3g'
    mol.verbose = 0
    mol.symmetry = False
    mol.cart = True
    mol.unit = "angstrom"
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 1000
    mf.kernel()

    # grad = mf.nuc_grad_method().kernel()
    # assert numpy.allclose(grad, 0.0, atol=1e-4)
    # hess = mf.Hessian().kernel()

    from eph.mol import rhf
    eph_an = rhf.ElectronPhononCoupling(mf)
    dv_ref = eph_an.kernel()

    eph_fd = ElectronPhononCoupling(mf)
    eph_fd.verbose = 0
    for stepsize in [1e-2, 1e-3, 1e-4, 1e-5]:
        # dv_sol = eph_fd.kernel(stepsize=stepsize)
        # err = numpy.linalg.norm(dv_ref - dv_sol)
        # numpy.savetxt(mol.stdout, dv_ref[0, 0][:10, :10], fmt="% 6.4e", delimiter=",", header="dv_ref")
        # numpy.savetxt(mol.stdout, dv_sol[0, 0][:10, :10], fmt="% 6.4e", delimiter=",", header="dv_sol")
        # print("stepsize = %6.4e, error = %6.4e" % (stepsize, err))

        from pyscf.eph.eph_fd import gen_moles, run_mfs
        mol1, mol2 = gen_moles(mol, stepsize * 0.5)
        mfs = run_mfs(mf, mol1, mol2)
        dv_sol_1 = get_vmat(mf, mfs, stepsize).reshape(dv_ref.shape)
        dv_sol_2 = eph_fd.kernel(stepsize=stepsize).reshape(dv_ref.shape)

        err = abs(dv_sol_1 - dv_sol_2).max()
        print("stepsize = %6.4e, error = %6.4e" % (stepsize, err))

        assert 1 == 2

        # dv_ref = dv_ref.reshape(dv_sol.shape)

        # err = abs(dv_ref - dv_sol).max()
        # print("stepsize = %6.4e, error = %6.4e" % (stepsize, err))

        # numpy.savetxt(mol.stdout, dv_ref[0][:10, :10], fmt="% 6.4e", delimiter=",", header="dv_ref")
        # numpy.savetxt(mol.stdout, dv_sol[0][:10, :10], fmt="% 6.4e", delimiter=",", header="dv_sol")
