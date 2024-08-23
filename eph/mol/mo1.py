import os, sys, tempfile
import numpy, scipy
from functools import reduce

import pyscf
from pyscf import lib, gto, scf, dft
from pyscf.scf import cphf

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

    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol=mol)
    ovlp_deriv = eph_obj.gen_ovlp_deriv(mol=mol)
    fock_deriv = eph_obj.gen_fock_deriv(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    vresp = eph_obj.base.gen_response(mo_coeff, mo_occ, hermi=1)

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

        s1 = ovlp_deriv(ia)
        h1_sol, v1 = fock_deriv(ia)
        f1 = h1_sol

        (m1_sol, e1), dm1 = eph_obj.solve_mo1(
            vresp, s1, f1, mo_energy, 
            mo_coeff, mo_occ, verbose=0
        )

        h1_err = abs(h1_sol - h1_ref).max()
        m1_err = abs(m1_sol - m1_ref).max()

        print("ia = %2d, h1_err = %6.4e, m1_err = %6.4e" % (ia, h1_err, m1_err))
        assert h1_err + m1_err < 1e-10

    print("All tests passed!")