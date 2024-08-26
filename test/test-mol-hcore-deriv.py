import os, sys
import numpy, scipy

import pyscf
from pyscf import gto, scf

import eph

def deriv_fd(mf, stepsize=1e-4):
    scan_obj = mf.as_scanner()
    dm0 = mf.make_rdm1()

    natm = mf.mol.natm
    nao = mf.mol.nao_nr()
    x0 = mf.mol.atom_coords(unit='Bohr')
    
    res = []
    for ix in range(natm * 3):
        ia, x = divmod(ix, 3)

        dx = numpy.zeros((natm, 3))
        dx[ia, x] = stepsize
        m1 = mf.mol.set_geom_(x0 + dx, unit='Bohr', inplace=False)
        scan_obj(m1)
        h1 = scan_obj.get_hcore()
        s1 = scan_obj.get_ovlp()
        d1 = scan_obj.make_rdm1()
        f1 = h1 + scan_obj.get_veff(m1, dm0)
        

        m2 = mf.mol.set_geom_(x0 - dx, unit='Bohr', inplace=False)
        scan_obj(m2)
        h2 = scan_obj.get_hcore()
        s2 = scan_obj.get_ovlp()
        d2 = scan_obj.make_rdm1()
        f2 = h2 + scan_obj.get_veff(m2, dm0)

        dh = (h1 - h2) / (2 * stepsize)
        ds = (s1 - s2) / (2 * stepsize)
        dd = (d1 - d2) / (2 * stepsize)
        df = (f1 - f2) / (2 * stepsize)

        res.append(
            {
                'dh': dh,
                'ds': ds,
                'dd': dd,
                'df': df
            }
        )

    return res
    
def deriv_an(mf, stepsize=1e-4):
    natm = mf.mol.natm
    nao = mf.mol.nao_nr()
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

    from eph.mol import rhf
    eph_obj = rhf.ElectronPhononCoupling(mf)
    eph_obj.verbose = 0
    fock_deriv = eph_obj.gen_fock_deriv(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ovlp_deriv = eph_obj.gen_ovlp_deriv(mol=mf.mol)
    hcor_deriv = eph_obj.gen_hcore_deriv(mol=mf.mol)

    res = []

    for ix in range(natm):
        s1 = ovlp_deriv(ix)
        f1, jk1 = fock_deriv(ix)
        h1 = hcor_deriv(ix)

        (t1, e1), d1 = eph_obj.solve_mo1(
            vresp, s1=s1, f1=f1,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
        )

        assert s1.shape == (3, nao, nao)
        assert f1.shape == (3, nao, nao)
        assert d1.shape == (3, nao, nao)
        assert d1.shape == (3, nao, nao)

        for x in range(3):
            res.append(
                {
                    'dh': h1[x],
                    'ds': s1[x],
                    'dd': d1[x],
                    'df': f1[x]
                }
            )

    return res

mol = gto.M()
mol.atom = '''
O       0.0000000000     0.0000000000     0.1146878262
H      -0.7540663886    -0.0000000000    -0.4587203947
H       0.7540663886    -0.0000000000    -0.4587203947
'''
mol.basis = 'sto3g' # 631g*'
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
mf.kernel()

fd = deriv_fd(mf) # .reshape(-1, nao, nao)
an = deriv_an(mf) # .reshape(-1, nao, nao)

for ix in range(natm * 3):
    r1 = fd[ix]
    r2 = an[ix]

    for k in r1.keys():
        v1 = r1[k]
        v2 = r2[k]
        assert numpy.allclose(v1, v2, atol=1e-6), f"{k} {ix} {v1} {v2}"

        print(f"\n{k = }, {ix = }, annalytical = ")
        numpy.savetxt(mf.stdout, v1, fmt='% 6.4f', delimiter=',')
        print(f"{k = }, {ix = }, finite difference = ")
        numpy.savetxt(mf.stdout, v2, fmt='% 6.4f', delimiter=',')
        err = abs(v1 - v2).max()
        print(f"{k = }, {ix = }, error = {err:6.4e}")

print("All tests passed")

