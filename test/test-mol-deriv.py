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
        s1 = scan_obj.get_ovlp()
        d1 = scan_obj.make_rdm1()
        v1 = m1.intor("int1e_nuc")
        j1, k1 = scan_obj.get_jk(dm0)
        f1 = h1 + scan_obj.get_veff(m1, dm0)
        veff1 = scan_obj.get_veff()
        eri1 = ao2mo.restore(1, scan_obj._eri, nao).reshape(nao, nao, nao, nao)
        assert eri1.shape == (nao, nao, nao, nao)

        fock1 = scan_obj.get_fock()

        m2 = mf.mol.set_geom_(x0 - dx, unit='Bohr', inplace=False)
        scan_obj(m2)
        h2 = scan_obj.get_hcore()
        s2 = scan_obj.get_ovlp()
        d2 = scan_obj.make_rdm1()
        v2 = m2.intor("int1e_nuc")
        j2, k2 = scan_obj.get_jk(dm0)
        f2 = h2 + scan_obj.get_veff(m2, dm0)
        veff2 = scan_obj.get_veff()
        eri2 = ao2mo.restore(1, scan_obj._eri, nao).reshape(nao, nao, nao, nao)
        assert eri2.shape == (nao, nao, nao, nao)

        fock2 = scan_obj.get_fock()

        dh = (h1 - h2) / (2 * stepsize)
        ds = (s1 - s2) / (2 * stepsize)
        dd = (d1 - d2) / (2 * stepsize)
        dveff = (veff1 - veff2) / (2 * stepsize)
        dfock = (fock1 - fock2) / (2 * stepsize)
        deri = (eri1 - eri2) / (2 * stepsize)
        deri = deri.reshape(nao * nao, nao * nao)

        res.append(
            {
                'dh': dh,
                'ds': ds,
                'dd': dd,
                'df': df,
                "dfock": dfock,
                "dveff": dveff
            }
        )

    return res

def gen_kine_deriv(mol):
    nao = mol.nao_nr()
    ipkin = mol.intor("int1e_ipkin")
    assert t1.shape == (3, nao, nao)

    def func(ia):
        s0, s1, p0, p1 = mol.aoslice_by_atom()[ia]
        t1 = numpy.zeros_like(ipkin)
        t1[:, p0:p1] -= ipkin[:, p0:p1]
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
    vnuc_deriv = eph_obj.gen_vnuc_deriv(mol=mf.mol)
    int2e_ip1 = mf.mol.intor('int2e_ip1').reshape(3, nao, nao, nao, nao)

    # check the permutation symmetry of int2e_ip1
    err = abs(int2e_ip1 - int2e_ip1.transpose(0, 1, 2, 4, 3)).max()
    assert err < 1e-10

    res = []

    for ia, (s0, s1, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
        s1 = ovlp_deriv(ia)
        h1 = hcor_deriv(ia)
        f1, jk1 = fock_deriv(ia)
        v1 = vnuc_deriv(ia)

        (t1, e1), d1 = eph_obj.solve_mo1(
            vresp, s1=s1, f1=f1,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
        )

        dfock = f1 + vresp(d1)

        # deri = numpy.zeros((3, nao, nao, nao, nao))
        # deri[:, p0:p1, :, :, :]  = int2e_ip1[:, p0:p1, :, :, :]
        # deri[:, :, p0:p1, :, :] += int2e_ip1[:, p0:p1, :, :, :].transpose(0, 2, 1, 3, 4)
        # deri[:, :, :, p0:p1, :] += int2e_ip1[:, p0:p1, :, :, :].transpose(0, 3, 4, 1, 2)
        # deri[:, :, :, :, p0:p1] += int2e_ip1[:, p0:p1, :, :, :].transpose(0, 3, 4, 2, 1)
        # deri = -deri.reshape(3, nao * nao, nao * nao)

        assert s1.shape == (3, nao, nao)
        assert f1.shape == (3, nao, nao)
        assert d1.shape == (3, nao, nao)
        assert d1.shape == (3, nao, nao)
        # assert deri.shape == (3, nao * nao, nao * nao)

        for x in range(3):
            res.append(
                {
                    'dh': h1[x],
                    'ds': s1[x],
                    'dd': d1[x],
                    'df': f1[x],
                    'dv': v1[x],
                    # 'deri': deri[x],
                    "dfock": dfock[x]
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
        if not (k in ["ds", "dh", "df", "dd", "dfock"]):
            continue
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

