import os, sys
import numpy, scipy

import pyscf
from pyscf.pbc import gto, scf

import eph

def deriv_fd(mf, stepsize=1e-4):
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
        assert mf1.converged

        h1 = mf1.get_hcore()
        from pyscf.pbc.scf.hf import get_t
        t1 = get_t(m1)
        v1 = h1 - t1

        s1 = mf1.get_ovlp()
        rho1 = mf1.make_rdm1()
        veff1 = mf1.get_veff()
        fock1 = mf1.get_fock()
        assert numpy.allclose(fock1, h1 + veff1)

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
        assert mf2.converged

        h2 = mf2.get_hcore()
        t2 = get_t(m2)
        v2 = h2 - t2

        s2 = mf2.get_ovlp()
        rho2 = mf2.make_rdm1()
        veff2 = mf2.get_veff()
        fock2 = mf2.get_fock()
        assert numpy.allclose(fock2, h2 + veff2)

        dh = (h1 - h2) / (2 * stepsize)
        dv = (v1 - v2) / (2 * stepsize)
        dt = (t1 - t2) / (2 * stepsize)

        ds = (s1 - s2) / (2 * stepsize)
        drho = (rho1 - rho2) / (2 * stepsize)
        dveff = (veff1 - veff2) / (2 * stepsize) + dv
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

def gen_kine_deriv(cell):
    nao = cell.nao_nr()
    ipkin = cell.pbc_intor("int1e_ipkin")
    assert ipkin.shape == (3, nao, nao)

    def func(ia):
        s0, s1, p0, p1 = cell.aoslice_by_atom()[ia]
        t1 = numpy.zeros_like(ipkin)
        t1[:, p0:p1, :] -= ipkin[:, p0:p1]
        t1[:, :, p0:p1] -= ipkin[:, p0:p1].transpose(0, 2, 1)
        return t1
    
    return func
    
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
    kine_deriv = gen_kine_deriv(mf.cell)

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

        fock1 = f1 + vresp(d1)
        t1 = kine_deriv(ia)
        v1 = h1 - t1
        veff1 = fock1 - t1

        for x in range(3):
            res.append(
                {
                    'dh': h1[x],
                    'ds': s1[x],
                    "drho": d1[x],
                    "dfock": fock1[x],
                    "dt": t1[x],
                    "dv": v1[x],
                    "dveff": veff1[x]
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
        if not (k in ["dh", "ds", "drho", "dfock", "dt", "dv", "dveff"]):
            continue

        v1 = r1[k]
        v2 = r2[k]
        err = abs(v1 - v2).max()

        v1 = v1[:10, :10]
        v2 = v2[:10, :10]

        if abs(v1).max() <= 1e-6:
            continue
        
        print(f"\n{k = :s}, {ix = }, error = {err:6.4e}")
        print(f"annalytical = ")
        numpy.savetxt(mf.stdout, v1, fmt='% 6.4f', delimiter=',')
        print(f"finite difference = ")
        numpy.savetxt(mf.stdout, v2, fmt='% 6.4f', delimiter=',')

        assert err < 1e-3, "Error too large"

print("All tests passed")

