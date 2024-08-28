import os, sys
import numpy, scipy

import pyscf
from pyscf import gto, scf, ao2mo

import eph

def deriv_fd(mf, stepsize=1e-4):
    # finite difference

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
        v1 = m1.intor("int1e_nuc")
        t1 = m1.intor("int1e_kin")
        assert numpy.allclose(h1, t1 + v1)

        s1 = scan_obj.get_ovlp()
        rho1 = scan_obj.make_rdm1()
        veff1 = scan_obj.get_veff()
        fock1 = scan_obj.get_fock()
        assert numpy.allclose(fock1, h1 + veff1)

        m2 = mf.mol.set_geom_(x0 - dx, unit='Bohr', inplace=False)
        scan_obj(m2)
        h2 = scan_obj.get_hcore()
        v2 = m2.intor("int1e_nuc")
        t2 = m2.intor("int1e_kin")
        assert numpy.allclose(h2, t2 + v2)

        s2 = scan_obj.get_ovlp()
        rho2 = scan_obj.make_rdm1()
        veff2 = scan_obj.get_veff()
        fock2 = scan_obj.get_fock()
        assert numpy.allclose(fock2, h2 + veff2)

        dh = (h1 - h2) / (2 * stepsize)
        dv = (v1 - v2) / (2 * stepsize)
        dt = (t1 - t2) / (2 * stepsize)

        ds = (s1 - s2) / (2 * stepsize)
        drho = (rho1 - rho2) / (2 * stepsize)
        dveff = (veff1 - veff2) / (2 * stepsize)
        dfock = (fock1 - fock2) / (2 * stepsize)

        res.append(
            {
                'dh': dh,
                'ds': ds,
                'drho': drho,
                'dv': dv,
                'dt': dt,
                'dveff': dveff,
                'dfock': dfock
            }
        )

    return res

def gen_kine_deriv(mol):
    nao = mol.nao_nr()
    ipkin = mol.intor("int1e_ipkin")
    assert ipkin.shape == (3, nao, nao)

    def func(ia):
        s0, s1, p0, p1 = mol.aoslice_by_atom()[ia]
        t1 = numpy.zeros_like(ipkin)
        t1[:, p0:p1, :] -= ipkin[:, p0:p1]
        t1[:, :, p0:p1] -= ipkin[:, p0:p1].transpose(0, 2, 1)
        return t1
    
    return func
    
def deriv_an(mf, dm0=None, stepsize=1e-4):
    natm = mf.mol.natm
    nao = mf.mol.nao_nr()
    nbas = mf.mol.nbas

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
    kine_deriv = gen_kine_deriv(mf.mol)
    int2e_ip1 = mf.mol.intor('int2e_ip1').reshape(3, nao, nao, nao, nao)

    # check the permutation symmetry of int2e_ip1
    err = abs(int2e_ip1 - int2e_ip1.transpose(0, 1, 2, 4, 3)).max()
    assert err < 1e-10

    res = []

    for ia, (s0, s1, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
        ds = ovlp_deriv(ia)
        dh = hcor_deriv(ia)
        df, djk1 = fock_deriv(ia)

        (t1, e1), drho = eph_obj.solve_mo1(
            vresp, s1=ds, f1=df,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
        )

        dfock = df + vresp(drho)
        dt = kine_deriv(ia)
        # dveff = dfock - 

        for x in range(3):
            res.append(
                {
                    'dh': dh[x],
                    'ds': ds[x],
                    'drho': drho[x],
                    'dt': dt[x],
                    # 'dveff': dveff[x],
                    'dfock': dfock[x]
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

    for k in r2.keys():
        v1 = r1[k]
        v2 = r2[k]
        err = abs(v1 - v2).max()

        v1 = v1[:10, :10]
        v2 = v2[:10, :10]

        print(f"\n{k = }, {ix = }, annalytical = ")
        numpy.savetxt(mf.stdout, v1, fmt='% 8.4f', delimiter=', ')
        print(f"{k = }, {ix = }, finite difference = ")
        numpy.savetxt(mf.stdout, v2, fmt='% 8.4f', delimiter=', ')
        
        print(f"{k = }, {ix = }, error = {err:6.4e}")

        assert err < 1e-6, "Error too large"

print("All tests passed")

