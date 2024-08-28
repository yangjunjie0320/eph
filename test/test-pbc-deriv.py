import os, sys
import numpy, scipy

import pyscf
from pyscf.pbc import gto, scf

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

        # the first geometry
        m1 = mf.mol.set_geom_(x0 + dx, unit='Bohr', inplace=False)
        m1.a = mf.mol.lattice_vectors()
        m1.ke_cutoff = mf.cell.ke_cutoff
        m1.build()

        mf1 = mf.__class__(m1)
        if hasattr(mf, "xc"):
            mf1.xc = mf.xc
        mf1.verbose = 0
        mf1.conv_tol = 1e-12
        mf1.conv_tol_grad = 1e-10
        mf1.max_cycle = 100
        mf1.exxdiv = None
        mf1.kernel(dm0=dm0)

        h1 = mf1.get_hcore()
        s1 = mf1.get_ovlp()
        d1 = mf1.make_rdm1()
        eri1 = mf1.with_df.get_eri(compact=False).reshape(nao * nao, nao * nao)
        j1 = numpy.einsum("ijkl,kl->ij", eri1.reshape(nao, nao, nao, nao), dm0)
        k1 = numpy.einsum("ikjl,kl->ij", eri1.reshape(nao, nao, nao, nao), dm0)
        f1 = h1 + j1 - 0.5 * k1
        fock1 = mf1.get_fock()
        assert numpy.allclose(f1, mf1.get_fock(dm=dm0))

        c1 = mf1.mo_coeff
        nao, nmo = c1.shape
        assert numpy.allclose(c1.T @ s1 @ c1, numpy.eye(nmo))

        # the second geometry
        m2 = mf.mol.set_geom_(x0 - dx, unit='Bohr', inplace=False)
        m2.a = mf.mol.lattice_vectors()
        m2.ke_cutoff = mf.cell.ke_cutoff
        m2.build()

        mf2 = mf.__class__(m2)
        if hasattr(mf, "xc"):
            mf2.xc = mf.xc
        mf2.verbose = 0
        mf2.conv_tol = 1e-12
        mf2.conv_tol_grad = 1e-10
        mf2.max_cycle = 100
        mf2.exxdiv = None
        mf2.kernel(dm0=dm0)

        h2 = mf2.get_hcore()
        s2 = mf2.get_ovlp()
        d2 = mf2.make_rdm1()
        eri2 = mf2.with_df.get_eri(compact=False).reshape(nao * nao, nao * nao)
        j2 = numpy.einsum("ijkl,kl->ij", eri2.reshape(nao, nao, nao, nao), dm0)
        k2 = numpy.einsum("ikjl,kl->ij", eri2.reshape(nao, nao, nao, nao), dm0)
        f2 = h2 + j2 - 0.5 * k2
        fock2 = mf2.get_fock()
        assert numpy.allclose(f2, mf2.get_fock(dm=dm0))

        c2 = mf2.mo_coeff
        nao, nmo = c2.shape
        assert numpy.allclose(c2.T @ s2 @ c2, numpy.eye(nmo))

        dh = (h1 - h2) / (2 * stepsize)
        ds = (s1 - s2) / (2 * stepsize)
        dd = (d1 - d2) / (2 * stepsize)
        df = (f1 - f2) / (2 * stepsize)
        dj = (j1 - j2) / (2 * stepsize)
        dk = (k1 - k2) / (2 * stepsize)
        deri = (eri1 - eri2) / (2 * stepsize)
        dfock = (fock1 - fock2) / (2 * stepsize)

        res.append(
            {
                'dh': dh,
                'ds': ds,
                'dd': dd,
                'deri': deri,
                'dj': dj,
                'dk': dk,
                'df': df,
                'dfock': dfock
            }
        )

    return res
    
def deriv_an(mf, stepsize=1e-4):
    natm = mf.mol.natm
    nbas = mf.mol.nbas

    nao = mf.mol.nao_nr()
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = mf.make_rdm1()

    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
    jk_sol = vresp(dm0)
    vj, vk = mf.get_jk(dm0)
    jk_ref = vj - 0.5 * vk

    err = abs(jk_sol - jk_ref).max()
    assert err < 1e-6, "Error too large: %6.4e" % err

    from eph.pbc import rhf
    eph_obj = rhf.ElectronPhononCoupling(mf)
    eph_obj.verbose = 0
    fock_deriv = eph_obj.gen_fock_deriv(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ovlp_deriv = eph_obj.gen_ovlp_deriv(mol=mf.mol)
    hcor_deriv = eph_obj.gen_hcore_deriv(mol=mf.mol)

    res = []

    for ia, (s0, s1, p0, p1) in enumerate(mf.mol.aoslice_by_atom()):
        s1 = ovlp_deriv(ia)
        f1, jk1 = fock_deriv(ia)
        h1 = hcor_deriv(ia)

        (t1, e1), d1 = eph_obj.solve_mo1(
            vresp, s1=s1, f1=f1,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
        )

        for x in range(3):
            res.append(
                {
                    'dh': h1[x],
                    'ds': s1[x],
                    "df": f1[x],
                    "dd": d1[x],
                    "dfock": f1[x] + vresp(d1[x])
                }
            )

    return res

from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
c = bulk("C", "diamond", a=3.5668)

from pyscf.pbc import gto, scf
cell = gto.Cell()
cell.atom = ase_atoms_to_pyscf(c)
cell.a = c.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.unit = 'A'
cell.verbose = 0
cell.ke_cutoff = 100
cell.exp_to_discard = 0.1
cell.build()

natm = cell.natm
nao = cell.nao_nr()

mf = scf.RHF(cell)
mf.verbose = 0
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-10
mf.max_cycle = 100
mf.exxdiv = None
mf.kernel(dm0=None)
dm0 = mf.make_rdm1()

fd = deriv_fd(mf) # .reshape(-1, nao, nao)
an = deriv_an(mf, stepsize=1e-4) # .reshape(-1, nao, nao)

for ix in range(natm * 3):
    r1 = fd[ix]
    r2 = an[ix]

    for k in r2.keys():
        # if k in ["dh", "ds", "df", "dd", "dfock"]:
        v1 = r1[k]
        v2 = r2[k]
        err = abs(v1 - v2).max()

        v1 = v1[:10, :10]
        v2 = v2[:10, :10]

        if abs(v1).max() <= 1e-6:
            continue

        print(f"{k = }, {ix = }, annalytical = ")
        numpy.savetxt(mf.stdout, v1, fmt='% 6.4f', delimiter=',')
        print(f"{k = }, {ix = }, finite difference = ")
        numpy.savetxt(mf.stdout, v2, fmt='% 6.4f', delimiter=',')
        
        print(f"{k = }, {ix = }, error = {err:6.4e}\n")

        assert err < 1e-3, "Error too large"

print("All tests passed")

