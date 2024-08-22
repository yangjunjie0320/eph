import os, sys, tempfile
import numpy, scipy
from functools import reduce

import pyscf
from pyscf import lib
from pyscf.scf import cphf

from jk import get_jk1, gen_hcore_deriv, get_ipovlp

# def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
#     from pyscf.grad import rks as rks_grad
#     from pyscf.dft import numint, gen_grid
#     from pyscf.hessian.rks import _make_dR_rho1

#     mol = hessobj.mol
#     mf = hessobj.base
#     if hessobj.grids is not None:
#         grids = hessobj.grids
#     else:
#         grids = mf.grids
#     if grids.coords is None:
#         grids.build(with_non0tab=True)

#     nao, nmo = mo_coeff.shape
#     ni = mf._numint
#     xctype = ni._xc_type(mf.xc)
#     aoslices = mol.aoslice_by_atom()
#     shls_slice = (0, mol.nbas)
#     ao_loc = mol.ao_loc_nr()
#     dm0 = mf.make_rdm1(mo_coeff, mo_occ)

#     v_ip = numpy.zeros((3,nao,nao))
#     vmat = numpy.zeros((mol.natm,3,nao,nao))
#     max_memory = max(2000, max_memory-vmat.size*8/1e6)
#     if xctype == 'LDA':
#         ao_deriv = 1
#         for ao, mask, weight, coords \
#                 in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
#             rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
#             vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
#             wv = weight * vxc[0]
#             aow = numint._scale_ao(ao[0], wv)
#             rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)

#             ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
#             wf = weight * fxc[0,0]
#             for ia in range(mol.natm):
#                 p0, p1 = aoslices[ia][2:]
# # First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
#                 rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
#                 wv = wf * rho1
#                 aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
#                 rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
#             ao_dm0 = aow = None

#     elif xctype == 'GGA':
#         ao_deriv = 2
#         for ao, mask, weight, coords \
#                 in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
#             rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
#             vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
#             wv = weight * vxc
#             wv[0] *= .5
#             rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

#             ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
#                       for i in range(4)]
#             wf = weight * fxc
#             for ia in range(mol.natm):
#                 dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
#                 wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
#                 wv[:,0] *= .5
#                 aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
#                 rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
#             ao_dm0 = aow = None

#     else:
#         raise NotImplementedError('meta-GGA')

#     for ia in range(mol.natm):
#         p0, p1 = aoslices[ia][2:]
#         # vmat[ia,:,p0:p1] += v_ip[:,p0:p1]
#         vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

#     return vmat

# def get_h1ao(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
#              atmlst=None, chkfile=None, verbose=None):
#     if chkfile is None:
#         chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
#         chkfile = chkfile.name
#     # assert os.path.exists(chkfile), '%s not found' % chkfile

#     mo_coeff = mo_coeff if mo_coeff is not None else mf_obj.mo_coeff
#     mo_occ = mo_occ if mo_occ is not None else mf_obj.mo_occ
#     mo_energy = mo_energy if mo_energy is not None else mf_obj.mo_energy
#     atmlst = atmlst if atmlst is not None else range(mf_obj.mol.natm)

#     if isinstance(mf_obj, pyscf.scf.hf.RHF): # restricted HF/KS
#         _rhf_h1ao(
#             mf_obj, mo_coeff=mo_coeff, 
#             mo_occ=mo_occ, mo_energy=mo_energy,
#             atmlst=atmlst, chkfile=chkfile
#         )
#         return chkfile

#     else:
#         raise NotImplementedError
    
# def solve_mo1(mf_obj, mo_coeff=None, mo_occ=None, mo_energy=None,
#               atmlst=None, chkfile=None, verbose=None, **kwargs):
#     if chkfile is None:
#         chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
#         chkfile = chkfile.name
#     # assert os.path.exists(chkfile), '%s not found' % chkfile

#     mo_coeff = mo_coeff if mo_coeff is not None else mf_obj.mo_coeff
#     mo_occ = mo_occ if mo_occ is not None else mf_obj.mo_occ
#     mo_energy = mo_energy if mo_energy is not None else mf_obj.mo_energy
#     atmlst = atmlst if atmlst is not None else range(mf_obj.mol.natm)

#     if isinstance(mf_obj, pyscf.scf.hf.RHF): # restricted HF/KS
#         _rhf_mo1(
#             mf_obj, mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy,
#             atmlst=atmlst, chkfile=chkfile, verbose=verbose, **kwargs

#         )
#         return chkfile

#     else:
#         raise NotImplementedError

def solve_mo1(mf_obj, ia=0, mo_coeff=None, mo_occ=None, mo_energy=None,
              atmlst=None, chkfile=None, verbose=None, conv_tol=1e-6,
              max_cycle=50, level_shift=0.0):
    log = lib.logger.new_logger(mf_obj, verbose=verbose)

    mol = mf_obj.mol
    aoslices = mol.aoslice_by_atom()
    nbas = mol.nbas
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    norb = nmo
    orbo = mo_coeff[:, mo_occ>0]
    nocc = orbo.shape[1]
    nvir = norb - nocc
    dm0 = mf_obj.make_rdm1(mo_coeff, mo_occ)

    from pyscf import dft
    if isinstance(mf_obj, dft.rks.KohnShamDFT):
        raise NotImplementedError

    else: # is Hartree-Fock
        from pyscf.scf import hf
        assert isinstance(mf_obj, hf.RHF)
        omega, alpha, hyb = 0.0, 0.0, 1.0
        is_hybrid = True
        vxc1 = None
        vxc2 = None
    
    assert abs(omega) < 1e-10

    hcore_deriv = gen_hcore_deriv(mf_obj)
    ipovlp = get_ipovlp(mf_obj)
    assert ipovlp.shape == (3, nao, nao)

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

    # h1 = numpy.asarray(h1).reshape(3, nao, nao)
    # s1 = numpy.asarray(s1).reshape(3, nao, nao)

    f1 = numpy.einsum("xmn,mp,ni->xpi", h1, mo_coeff, orbo, optimize=True)
    s1 = numpy.einsum("xmn,mp,ni->xpi", s1, mo_coeff, orbo, optimize=True)
    f1 = f1.reshape(3, nmo, nocc)
    s1 = s1.reshape(3, nmo, nocc)

    z1, e1 = cphf.solve(
        func, mo_energy, mo_occ, f1, s1,
        tol=1e-20,
        max_cycle=200,
        level_shift=level_shift
    )

    t1 = numpy.einsum('mq,xqi->xmi', mo_coeff, z1).reshape(-1, nao, nocc)
    dm1 = 2.0 * numpy.einsum('xmi,ni->xmn', t1, orbo)
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    dm1 = dm1.reshape(3, nao, nao)

    return vj1, vk1, dm1, h1, t1

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

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    # mf.xc = "PBE0"
    mf.kernel()

    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff

    from rhf import ElectronPhononCoupling
    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR).name
    eph_obj.verbose = 0
    # eph_obj.solve_mo1(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    # res_sol = eph_obj.chkfile

    res_ref = mf.Hessian().make_h1(
        mo_coeff, mo_occ,
        chkfile=mf.chkfile, 
        atmlst=range(natm),
        verbose=10
    )

    mf.conv_tol_cpscf = 1e-20
    hess_obj = mf.Hessian()
    hess_obj.max_cycle = 200


    for ia in range(natm):
        hess_obj.solve_mo1(
            mo_energy, mo_coeff, mo_occ,
            mf.chkfile, atmlst=[ia],
            verbose=10, max_memory=0
        )

        h1_ref = lib.chkfile.load(res_ref, 'scf_f1ao/%d' % ia)
        m1_ref = lib.chkfile.load(res_ref, 'scf_mo1/%d' % ia)

        h1_sol, m1_sol = solve_mo1(
            mf, ia=ia, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            atmlst=None, chkfile=None, verbose=10
        )[-2:]

        h1_err = abs(h1_sol - h1_ref).max()
        m1_err = abs(m1_sol - m1_ref).max()

        print("ia = %2d, h1_err = %6.4e, m1_err = %6.4e" % (ia, h1_err, m1_err))
        assert h1_err + m1_err < 1e-10

    print("All tests passed!")