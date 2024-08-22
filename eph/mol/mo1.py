import os, sys, tempfile
import numpy, scipy
from functools import reduce

import pyscf
from pyscf import lib
from pyscf.scf import cphf

from jk import get_jk1, gen_hcore_deriv, get_ipovlp

def get_h1ao(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
             atmlst=None, chkfile=None, verbose=None):
    if chkfile is None:
        chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = chkfile.name
    # assert os.path.exists(chkfile), '%s not found' % chkfile

    mo_coeff = mo_coeff if mo_coeff is not None else mf_obj.mo_coeff
    mo_occ = mo_occ if mo_occ is not None else mf_obj.mo_occ
    mo_energy = mo_energy if mo_energy is not None else mf_obj.mo_energy
    atmlst = atmlst if atmlst is not None else range(mf_obj.mol.natm)

    if isinstance(mf_obj, pyscf.scf.hf.RHF): # restricted HF/KS
        _rhf_h1ao(
            mf_obj, mo_coeff=mo_coeff, 
            mo_occ=mo_occ, mo_energy=mo_energy,
            atmlst=atmlst, chkfile=chkfile
        )
        return chkfile

    else:
        raise NotImplementedError
    
def solve_mo1(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
              atmlst=None, chkfile=None, verbose=None, **kwargs):
    if chkfile is None:
        chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = chkfile.name
    # assert os.path.exists(chkfile), '%s not found' % chkfile

    mo_coeff = mo_coeff if mo_coeff is not None else mf_obj.mo_coeff
    mo_occ = mo_occ if mo_occ is not None else mf_obj.mo_occ
    mo_energy = mo_energy if mo_energy is not None else mf_obj.mo_energy
    atmlst = atmlst if atmlst is not None else range(mf_obj.mol.natm)

    if isinstance(mf_obj, pyscf.scf.hf.RHF): # restricted HF/KS
        _rhf_mo1(
            mf_obj, mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy,
            atmlst=atmlst, chkfile=chkfile, verbose=verbose, **kwargs

        )
        return chkfile

    else:
        raise NotImplementedError

def _rhf_h1ao(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
              atmlst=None, chkfile=None):
    mol = mf_obj.mol
    aoslices = mol.aoslice_by_atom()
    nbas = mol.nbas
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    orbo = mo_coeff[:, mo_occ>0]
    nocc = orbo.shape[1]
    nvir = nmo - nocc
    dm0 = mf_obj.make_rdm1(mo_coeff, mo_occ)

    from pyscf import dft
    if isinstance(mf_obj, dft.rks.KohnShamDFT):
        # test if the functional has the second derivative
        ni = mf_obj._numint
        ni.libxc.test_deriv_order(
            mf_obj.xc, 2, raise_error=True
        )

        from pyscf.hessian.rks import Hessian
        from pyscf.hessian.rks import _get_vxc_deriv1
        vxc1 = _get_vxc_deriv1(
            Hessian(mf_obj), mo_coeff=mo_coeff, 
            mo_occ=mo_occ, max_memory=mf_obj.max_memory,
        )

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(
            mf_obj.xc, spin=mol.spin
        )

        is_hybrid = ni.libxc.is_hybrid_xc(mf_obj.xc)

    else: # is Hartree-Fock
        assert isinstance(mf_obj, scf.hf.RHF)
        omega, alpha, hyb = 0.0, 0.0, 1.0
        is_hybrid = True
        vxc1 = None
    
    assert abs(omega) < 1e-10

    hcore_deriv = gen_hcore_deriv(mf_obj)
    ipovlp = get_ipovlp(mf_obj)
    assert ipovlp.shape == (3, nao, nao)

    # add for the case of KS object
    for i0, ia in enumerate(atmlst):
        s0, s1, p0, p1 = aoslices[ia]

        shls_slice = (s0, s1) + (0, nbas) * 3
        script_dms  = ['ji->s1kl', -dm0[:, p0:p1]] # vj1
        script_dms += ['lk->s1ij', -dm0]           # vj2
        script_dms += ['li->s1kj', -dm0[:, p0:p1]] # vk1
        script_dms += ['jk->s1il', -dm0]           # vk2

        if is_hybrid:
            res = get_jk1(
                mf_obj, script_dms=script_dms,
                shls_slice=shls_slice
            )

            vj1, vj2, vk1, vk2 = res
            vjk = vj1 - vk1 * 0.5 * hyb
            vjk[:, p0:p1] += vj2 - vk2 * 0.5 * hyb
        else:
            res = get_jk1(
                mf_obj, script_dms=script_dms[:4],
                shls_slice=shls_slice
            )

            vj1, vj2 = res
            vjk = vj1
            vjk[:, p0:p1] += vj2

            vk1, vk2 = None, None
        
        h1 = vjk + vjk.transpose(0, 2, 1)
        h1 += hcore_deriv(ia)
        if vxc1 is not None:
            h1 += vxc1[ia]

        s1 = numpy.zeros_like(ipovlp)
        s1[:, p0:p1, :] -= ipovlp[:, p0:p1]
        s1[:, :, p0:p1] -= ipovlp[:, p0:p1].transpose(0, 2, 1)

        key = 'scf_f1ao/%d' % ia # is this term d F / d R?
        lib.chkfile.save(chkfile, key, h1)

        key = 'scf_s1ao/%d' % ia
        lib.chkfile.save(chkfile, key, s1)

        key = 'scf_j1ao/%d' % ia
        lib.chkfile.save(chkfile, key, vj1)

        key = 'scf_j2ao/%d' % ia
        lib.chkfile.save(chkfile, key, vj2)

        if is_hybrid:
            key = 'scf_k1ao/%d' % ia
            lib.chkfile.save(chkfile, key, vk1)

            key = 'scf_k2ao/%d' % ia
            lib.chkfile.save(chkfile, key, vk2)

        if vxc1 is not None:
            key = 'scf_vxc1/%d' % ia
            lib.chkfile.save(chkfile, key, vxc1[ia])

def _rhf_mo1(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
             atmlst=None, chkfile=None, verbose=None, 
             level_shift=0.0, conv_tol=1e-8, max_cycle=50):
    log = pyscf.lib.logger.new_logger(mf_obj, verbose)

    mol = mf_obj.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    norb = nmo
    orbo = mo_coeff[:, mo_occ>0]
    nocc = orbo.shape[1]
    nvir = nmo - nocc
    dm0 = mf_obj.make_rdm1(mo_coeff, mo_occ)

    assert hasattr(mf_obj, 'gen_response')
    vresp = mf_obj.gen_response(mo_coeff, mo_occ, hermi=1)

    def func(t1):
        t1 = t1.reshape(-1, norb, nocc)
        dm1 = 2.0 * numpy.einsum('xpi,mp,ni->xmn', t1, mo_coeff, orbo)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        v1 = vresp(dm1)
        v1vo = numpy.einsum("xmn,mp,ni->xpi", v1, mo_coeff, orbo, optimize=True)
        return v1vo

    max_memory  = mf_obj.max_memory * 0.8
    max_memory -= lib.current_memory()[0]
    max_memory  = max(2000, max_memory)
    max_memory *= (1e6 / 8) # convert to bytes

    size = nmo * nocc * 3 * 6
    blksize = int(max_memory / size)
    blksize = max(2, blksize)
    blksize = min(len(atmlst), blksize)
    log.info(
        'nao = %d, nmo = %d, nocc = %d, nvir = %d, blksize = %d',
        nao, nmo, nocc, nvir, blksize
    )
    log.info(
        'max_memory %d MB (current use %d MB)',
        mf_obj.max_memory, lib.current_memory()[0]
    )

    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        size = ia1 - ia0

        assert os.path.exists(chkfile), '%s not found' % chkfile
        h1 = [lib.chkfile.load(chkfile, 'scf_f1ao/%d' % ia) for ia in atmlst[ia0:ia1]]
        s1 = [lib.chkfile.load(chkfile, 'scf_s1ao/%d' % ia) for ia in atmlst[ia0:ia1]]
        h1 = numpy.asarray(h1).reshape(size, 3, nao, nao)
        s1 = numpy.asarray(s1).reshape(size, 3, nao, nao)

        h1 = numpy.einsum("axmn,mp,ni->axpi", h1, mo_coeff, orbo, optimize=True)
        s1 = numpy.einsum("axmn,mp,ni->axpi", s1, mo_coeff, orbo, optimize=True)
        h1 = h1.reshape(size * 3, nmo, nocc)
        s1 = s1.reshape(size * 3, nmo, nocc)

        m1, e1 = cphf.solve(
            func, mo_energy, mo_occ, h1, s1,
            tol=conv_tol * size,
            max_cycle=max_cycle,
            level_shift=level_shift
        )

        m1 = numpy.einsum('mq,xqi->xmi', mo_coeff, m1).reshape(size, 3, nao, nocc)
        e1 = e1.reshape(size, 3, nocc, nocc)

        for ia in atmlst[ia0:ia1]:
            key = 'scf_mo1/%d' % ia
            lib.chkfile.save(chkfile, key, m1[ia])

            key = 'scf_e1/%d' % ia
            lib.chkfile.save(chkfile, key, e1[ia])

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

    mf = scf.RKS(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.xc = "PBE0"
    mf.kernel()

    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff

    from rhf import ElectronPhononCoupling
    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR).name
    eph_obj.verbose = 0
    eph_obj.solve_mo1(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    res_sol = eph_obj.chkfile

    res_ref = mf.Hessian().make_h1(
        mo_coeff, mo_occ,
        chkfile=mf.chkfile, 
        atmlst=range(natm),
        verbose=10
    )

    mf.Hessian().solve_mo1(
        mo_energy, mo_coeff, mo_occ,
        mf.chkfile, atmlst=range(natm),
        verbose=10
    )

    for ia in range(natm):
        h1_sol = lib.chkfile.load(res_sol, 'scf_f1ao/%d' % ia)
        h1_ref = lib.chkfile.load(res_ref, 'scf_f1ao/%d' % ia)
        h1_err = abs(h1_sol - h1_ref).max()

        m1_sol = lib.chkfile.load(res_sol, 'scf_mo1/%d' % ia)
        m1_ref = lib.chkfile.load(res_ref, 'scf_mo1/%d' % ia)
        m1_err = abs(m1_sol - m1_ref).max()

        print("ia = %2d, h1_err = %6.4e, m1_err = %6.4e" % (ia, h1_err, m1_err))
        assert h1_err + m1_err < 1e-10

    print("All tests passed!")