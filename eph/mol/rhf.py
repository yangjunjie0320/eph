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
                    max_memory=4000, verbose=None):
    log = logger.new_logger(eph_obj, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol_obj = eph_obj.mol
    scf_obj = eph_obj.base

    if mo_energy is None:
        mo_energy = scf_obj.mo_energy
    
    if mo_coeff is None:
        mo_coeff = scf_obj.mo_coeff
    
    if mo_occ is None:
        mo_occ = scf_obj.mo_occ

    nao, nmo = mo_coeff.shape

    # Check if the mo1 is provided
    mo1_to_save = eph_obj.chkfile
    if mo1_to_save is not None:
        log.debug('mo1 will be saved in %s', mo1_to_save)

    h1, mo1 = eph_obj.solve_mo1(
        mo_energy, mo_coeff, mo_occ,
        tmpfile=mo1_to_save,
        log=log
    )

    t1 = log.timer('solving CP-SCF equations', *t0)
    dvnuc = eph_obj.gen_vnuc_deriv(mol_obj)
    dveff = eph_obj.gen_veff_deriv(mo_occ, mo_coeff, scf_obj=scf_obj, c1=mo1, h1=h1, log=log)

    dv = [dvnuc(ia) + dveff(ia) for ia in range(mol_obj.natm)]
    dv = numpy.asarray(dv)
    t1 = log.timer('dv', *t0)

    return dv.reshape(-1, nao, nao)

def make_h1(mo_energy, mo_coeff, mo_occ, tmpfile=None, scf_obj=None, log=None):
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

def gen_veff_deriv(mo_occ, mo_coeff, scf_obj=None, c1=None, h1=None, log=None):
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
        assert c1 is not None
        if isinstance(c1, str):
            assert os.path.exists(c1), '%s not found' % c1
            t1 = lib.chkfile.load(c1, 'scf_mo1/%d' % ia)
            t1 = t1.reshape(-1, nao, nocc)

        else:
            t1 = c1[ia].reshape(-1, nao, nocc)

        assert h1 is not None
        if isinstance(h1, str):
            assert os.path.exists(h1), '%s not found' % h1
            vj1 = lib.chkfile.load(h1, 'eph_vj1ao/%d' % ia)
            vk1 = lib.chkfile.load(h1, 'eph_vk1ao/%d' % ia)
        
        else:
            vj1 = h1[ia].vj1
            vk1 = h1[ia].vk1

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
    
    def _write(self, eph, verbose=None):
        '''Format output of electron-phonon coupling tensor.

        Args:
            mol : Mole
                Molecule object.
            eph : 3-d array
                Electron-phonon coupling tensor, the shape is (natm * 3, norb, norb)
            atmlst : list
                List of atom indices.
        '''

        log = logger.new_logger(mol, verbose)
        natm = mol.natm
        nao = mol.nao_nr()
        eph = eph.reshape(-1, nao, nao)
        nmode = eph.shape[0]

        for imode in range(nmode):
            eph_i = eph[imode]
            info = 'electron-phonon coupling for mode %d' % imode
            title = " EPH-%s " % self.__class__.__name__
            l = max(20, (len(info) - len(title)) // 2)

            log.info("\n" + "-" * l + title + "-" * l)
            log.info(info)

            from pyscf.tools.dump_mat import dump_rec
            dump_rec(log, eph_i, label=mol.ao_labels(), start=0, tol=1e-8)
    
    def _finalize(self):
        assert self.eph is not None
        self._write(self.eph, self.verbose)
    
    
    def gen_vnuc_deriv(self, mol):
        return gen_vnuc_deriv(mol)
    
    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, c1=None, h1=None, log=None):
        raise NotImplementedError
    
    def solve_mo1(self, *args, **kwargs):
        raise NotImplementedError
    
    def kernel(self, *args, **kwargs):
        dv = kernel(self, *args, **kwargs)
        # self.eph = eph
        # self._finalize()
        return dv

class RHF(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    # TODO: add the arguments
    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, tmpfile=None, log=None):
        h1 = make_h1(mo_energy, mo_coeff, mo_occ, tmpfile=tmpfile, scf_obj=self.base, log=log)
        c1 = hessian.rhf.solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1, fx=None)[0]
        return h1, c1
    
    def gen_veff_deriv(self, mo_occ, mo_coeff, scf_obj=None, c1=None, h1=None, log=None):
        return gen_veff_deriv(mo_occ, mo_coeff, scf_obj=scf_obj, c1=c1, h1=h1, log=log)
    
if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = '''
    O   -0.0171307   -0.0257390    1.7511303
    H    0.0107978    0.8956887    1.3895372
    H    0.6941101   -0.5072723    1.2583225
    O    0.0172522   -0.0257742   -1.7509868
    H   -0.0108795    0.8957342   -1.3896115
    H   -0.6940451   -0.5072835   -1.2582398
    '''
    mol.unit  = 'bohr'
    mol.basis = 'sto3g'
    mol.verbose = 0
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
    # eph_obj.chkfile = None
    eph_sol = eph_obj.kernel()
    chkfile = eph_obj.chkfile
    assert 1 == 2

    from pyscf.eph.rhf import EPH
    from pyscf.eph.rhf import get_eph
    def func(*args):
        print("hack _freq_mass_weighted_vec")
        print(args)
        return numpy.eye(natm * 3)
    pyscf.eph.rhf._freq_mass_weighted_vec = func
    eph_obj = EPH(mf)
    eph_ref = get_eph(eph_obj, chkfile, omega=None, vec=None, mo_rep=False)

    err = numpy.linalg.norm(eph_sol - eph_ref)
    print('eph error %6.4e' % err)
