import numpy, scipy

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
                    mo1=None, max_memory=4000, verbose=None):
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

    # Check if the mo1 is provided
    tmpfile = lib.chkfile.newfile(mol_obj, 'eph_tmpfile')
    if mo1 is None:
        eph_obj.make_h1(
            mo_coeff, mo_occ, 
            eph_obj.chkfile, 
            None, log
        )

        mo1, mo_e1 = eph_obj.solve_mo1(
            mo_energy, mo_coeff, mo_occ, 
            eph_obj.chkfile, None, None, 
            max_memory, log,
        )

        t1 = log.timer('solving mo1 eq', *t0)

    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1 = {int(k): mo1[k] for k in mo1}
    assert isinstance(mo1, dict)

    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ > 0]

    dvnuc = eph_obj.gen_vnuc_deriv()
    dveff = eph_obj.gen_veff_deriv(mo_occ, mo_coeff)
    dvind = eph_obj.gen_vind_deriv(mo_occ, mo_coeff)

    dv = []
    for ia in range(natm):
        dv.append(dvnuc(ia) + dvind(ia) + dveff(ia))

    dv = numpy.asarray(dv)
    t1 = log.timer('dv', *t0)

    return dv.reshape(-1, nao, nao)

def gen_vjk_deriv(mo_occ, mo_coeff, scf_obj=None):
    aoslice_by_atom = scf_obj.mol.aoslice_by_atom()
    nbas = scf_obj.mol.nbas

    orbo = mo_coeff[:, mo_occ > 0]
    dm = numpy.dot(orbo, orbo.T) * 2

    def vjk_deriv(ia):
        s0, s1, p0, p1 = aoslice_by_atom[ia]


        script_dms = ['ji->s2kl', -dm[:, p0:p1], 'li->s1kj', -dm[:, p0:p1]]
        shls_slice = (s0, s1) + (0, nbas) * 3

        from pyscf.hessian import rhf
        vj1, vk1 = rhf._get_jk(
            scf_obj.mol, 'int2e_ip1', 3, 's2kl',
            script_dms=script_dms,
            shls_slice=shls_slice
            )

        vjk1 = vj1 - vk1 * 0.5
        return vjk1 + vjk1.transpose(0, 2, 1)
    
    return vjk_deriv

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
        eph = eph.reshape(natm, 3, nao, nao)

        for m in range(nao):
            for n in range(nao):
                eph_mn = eph[:, :, m, n]

                info = 'electron-phonon coupling for (%2d, %2d)' % (m, n)
                title = " EPH-%s " % self.__class__.__name__
                l = (len(info) - len(title)) // 2 
                l = max(l, 20)

                log.info("\n" + "-" * l + title + "-" * l)
                log.info(info)
                for ia in range(natm):
                    log.info(
                        "%2d %2s % 12.8f % 12.8f % 12.8f",
                        ia, mol.atom_symbol(ia), *eph_mn[ia, :]
                        )
                log.info("-" * (len(title) + 2 * l))
    
    def _finalize(self):
        assert self.eph is not None
        self._write(self.eph, self.verbose)
    
    def gen_vnuc_deriv(self):
        mol = self.mol
        def vnuc_deriv(atm_id):
            with mol.with_rinv_at_nucleus(atm_id):
                vrinv  = mol.intor('int1e_iprinv', comp=3)
                vrinv *= -mol.atom_charge(atm_id)
            return vrinv + vrinv.transpose(0, 2, 1)
        return vnuc_deriv
    
    def gen_vind_deriv(self, mo_occ, mo_coeff):
        raise NotImplementedError
    
    def gen_veff_deriv(self, mo_occ, mo_coeff):
        raise NotImplementedError
    
    def solve_mo1(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_h1(self, *args, **kwargs):
        raise NotImplementedError
    
    def kernel(self, *args, **kwargs):
        eph = kernel(self, *args, **kwargs)
        self.eph = eph

        self._finalize()
        return eph

class RHF(ElectronPhononCouplingBase):
    def __init__(self, method):
        assert isinstance(method, scf.hf.RHF)
        ElectronPhononCouplingBase.__init__(self, method)

    # TODO: add the arguments
    def solve_mo1(self, *args, **kwargs):
        return hessian.rhf.solve_mo1(self.base, *args, **kwargs)

    def make_h1(self, mo_coeff, mo_occ, tmpfile=None, log=None):
        mol = self.mol

        nao, nmo = mo_coeff.shape
        mask = mo_occ > 0
        orbo = mo_coeff[:, mask]
        dm0 = numpy.dot(orbo, orbo.T) * 2
        hcore_deriv = self.base.nuc_grad_method().hcore_generator(mol)

        aoslices = mol.aoslice_by_atom()
        h1ao = [None] * mol.natm
        for i0, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, mol.nbas)*3
            vj1, vj2, vk1, vk2 = _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                        'lk->s1ij', -dm0         ,  # vj2
                                        'li->s1kj', -dm0[:,p0:p1],  # vk1
                                        'jk->s1il', -dm0         ], # vk2
                                        shls_slice=shls_slice)
            vhf = vj1 - vk1*.5
            vhf[:,p0:p1] += vj2 - vk2*.5
            h1 = vhf + vhf.transpose(0,2,1)
            h1 += hcore_deriv(ia)

            if chkfile is None:
                h1ao[ia] = h1
            else:
                key = 'scf_f1ao/%d' % ia
                lib.chkfile.save(chkfile, key, h1)
        if chkfile is None:
            return h1ao
        else:
            return chkfile

    def gen_vind_deriv(self, mo_occ, mo_coeff):
        nao, nmo = mo_coeff.shape
        mask = mo_occ > 0
        orbo = mo_coeff[:, mask]
        nocc = orbo.shape[1]

        from pyscf.scf._response_functions import _gen_rhf_response
        vresp = _gen_rhf_response(mf, mo_coeff, mo_occ, hermi=1)

        def vind(ia):
            mo1 = lib.chkfile.load(self.chkfile, 'scf_mo1/%d' % ia)
            mo1 = mo1.reshape(-1, nao, nocc)

            dm1 = 2.0 * numpy.einsum('xpi,qi->xpq', mo1, orbo)
            dm1 = dm1 + dm1.transpose(0, 2, 1)
            v1 = vresp(dm1)
            return v1
        
        return vind
    
    def gen_veff_deriv(self, mo_occ, mo_coeff):
        return gen_vjk_deriv(mo_occ, mo_coeff, self.base)
    
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
    eph_sol = eph_obj.kernel()
    chkfile = eph_obj.chkfile

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
