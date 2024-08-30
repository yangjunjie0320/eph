import os, sys
import numpy, scipy

import pyscf
from pyscf import gto, scf, ao2mo

import eph

def gen_vnuc_deriv(mol):
    # the matrix element derivative
    # d/dx < mu | v_nuc | nu >
    ipnuc = mol.intor("int1e_ipnuc")

    def func(ia):
        s0, s1, p0, p1 = mol.aoslice_by_atom()[ia]

        with mol.with_rinv_at_nucleus(ia):
            dv =  mol.intor("int1e_iprinv", comp=3)
            dv *= -mol.atom_charge(ia)

            dv[:, p0:p1] -= ipnuc[:, p0:p1]
            return dv + dv.transpose(0, 2, 1).conj()

    return func

def gen_unuc_deriv(mol):
    # the matrix element of the operator derivative
    # < mu | d/dx v_nuc | nu >

    def func(ia):
        s0, s1, p0, p1 = mol.aoslice_by_atom()[ia]

        with mol.with_rinv_at_nucleus(ia):
            du =  mol.intor("int1e_iprinv", comp=3)
            du *= -mol.atom_charge(ia)

            return du + du.transpose(0, 2, 1).conj()

    return func

def solve(mf=None, u=None):
    mol = mf.mol
    nao = mol.nao_nr()
    nbas = mol.nbas

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    orb = mo_coeff
    orbo = mo_coeff[:, mo_occ > 0]
    orbv = mo_coeff[:, mo_occ == 0] 

    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    nao, nmo = mo_coeff.shape
    norb = nocc + nvir
    assert norb == nmo

    assert u.shape == (nao, nao)
    uvo = numpy.einsum("mn,mp,ni->pi", u, orbv, orbo)
    assert uvo.shape == (nvir, nocc)

    vresp = mf.gen_response(singlet=None, hermi=1)

    def func(xvo):
        xvo = xvo.reshape(-1, nvir, nocc)
        xao = numpy.einsum("xai,ma,ni->xmn", xvo, orbv, orbo, optimize=True) * 2.0
        v = vresp(xao + xao.transpose(0, 2, 1))
        return numpy.einsum("xmn,ma,ni->xai", v, orbv, orbo, optimize=True)

    # solve cphf equation
    from pyscf.scf import cphf
    tvo = cphf.solve(
        func, mo_energy, mo_occ, uvo,
        max_cycle=50, tol=1e-8,
    )[0]
    assert tvo.shape == (nvir, nocc)
    return tvo

def deriv_fd(mf, stepsize=1e-4):
    mol = mf.mol
    natm = mf.mol.natm
    symbs = [mf.mol.atom_pure_symbol(ia) for ia in range(natm)]
    x0 = mf.mol.atom_coords(unit='Bohr')

    basis = mf.mol.basis
    nao = mf.mol.nao_nr()

    hcore = mf.get_hcore()
    kine = mol.intor("int1e_kin")
    vnuc = mol.intor("int1e_nuc")
    coeff = mf.mo_coeff
    ovlp = mf.get_ovlp()
    assert numpy.allclose(hcore, kine + vnuc), abs(hcore - kine - vnuc).max()

    vresp = mf.gen_response(singlet=None, hermi=1)
    
    res = []
    for ix in range(natm * 3):

        ia, x = divmod(ix, 3)

        dx = numpy.zeros_like(x0)
        dx[ia, x] += stepsize
        
        atom  = [(s, x0[ia] + dx[ia]) for ia, s in enumerate(symbs)]
        atom += [("X:" + s,   x0[ia]) for ia, s in enumerate(symbs)]

        m1 = gto.M(
            verbose=0, unit = "Bohr",
            atom = atom, basis = basis
        )
        int1e_nuc = m1.intor("int1e_nuc")
        assert int1e_nuc.shape == (2 * nao, 2 * nao)

        v1 = int1e_nuc[:nao, :nao] # in
        u1 = int1e_nuc[nao:, nao:]

        mf.get_hcore = lambda *args: kine + u1
        mf.kernel()
        assert mf.converged

        dm1 = mf.make_rdm1()
        fock1 = mf.get_fock(dm=dm1)

        atom  = [(s, x0[ia] - dx[ia]) for ia, s in enumerate(symbs)]
        atom += [("X:" + s,   x0[ia]) for ia, s in enumerate(symbs)]

        m2 = gto.M(
            verbose=0, unit = "Bohr",
            atom = atom, basis = basis
        )
        int1e_nuc = m2.intor("int1e_nuc")
        assert int1e_nuc.shape == (2 * nao, 2 * nao)

        v2 = int1e_nuc[:nao, :nao] # in
        u2 = int1e_nuc[nao:, nao:]

        mf.get_hcore = lambda *args: kine + u2
        mf.kernel()
        assert mf.converged

        dm2 = mf.make_rdm1()
        fock2 = mf.get_fock(dm=dm2)

        dm1_mo = coeff.T @ ovlp @ dm1 @ ovlp @ coeff
        dm2_mo = coeff.T @ ovlp @ dm2 @ ovlp @ coeff

        dv = (v1 - v2) / (2 * stepsize)
        du = (u1 - u2) / (2 * stepsize)
        dfock = (fock1 - fock2) / (2 * stepsize)
        drho = (dm1 - dm2) / (2 * stepsize)

        ddm_mo = (dm1_mo - dm2_mo) / (2.0 * stepsize)

        res.append(
            {
                'dv': dv,
                'du': du,
                'dfock': dfock,
                # 'drho': drho,
                # 'dm1_mo': dm1_mo,
                # 'dm2_mo': dm2_mo,
                'ddm_mo': ddm_mo,
            }
        )

    return res

def deriv_an(mf, stepsize=1e-4):
    mol = mf.mol
    nao = mol.nao_nr()
    nbas = mol.nbas

    unuc_deriv = gen_unuc_deriv(mol)
    vnuc_deriv = gen_vnuc_deriv(mol)

    coeff = mf.mo_coeff
    vresp = mf.gen_response(singlet=None, hermi=1)

    res = []
    for ia in range(mol.natm):
        du = unuc_deriv(ia)
        dv = vnuc_deriv(ia)

        for x in range(3):
            t = solve(mf, u=(du[x]))
            nocc = t.shape[1]

            drho = numpy.zeros_like(du[0])
            drho[nocc:, :nocc] += t
            drho += drho.T
            drho *= 2.0

            res.append(
                {
                    'du': du[x],
                    'dv': dv[x],
                    'ddm_mo': drho,
                    'dfock': du[x] + vresp(coeff @ drho @ coeff.T)
                }
            )

    return res

if __name__ == '__main__':
    for stepsize in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        from pyscf import gto, scf
        mol = gto.M()
        mol.atom = '''
        O       0.0000000000     0.0000000000     0.1146878262
        H      -0.7540663886    -0.0000000000    -0.4587203947
        H       0.7540663886    -0.0000000000    -0.4587203947
        '''
        mol.basis = 'sto3g' # 631g*'
        mol.verbose = 0
        mol.symmetry = False
        mol.cart = False
        mol.unit = "AA"
        mol.build()

        natm = mol.natm
        nao = mol.nao_nr()
        
        print(f"\nstepsize = {stepsize:6.4e}")
        mf = scf.RHF(mol.copy())
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-12
        mf.max_cycle = 1000
        mf.kernel()

        from pyscf.eph.eph_fd import gen_moles, run_mfs, get_vmat
        mol_a, mol_b = gen_moles(mol, stepsize / 2.0)
        mfs = run_mfs(mf, mol_a, mol_b)
        vmat = get_vmat(mf, mfs, disp=stepsize)

        coeff = mf.mo_coeff
        vresp = mf.gen_response(singlet=None, hermi=1)

        mf = scf.RHF(mol.copy())
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-12
        mf.max_cycle = 1000
        mf.kernel()

        an = deriv_an(mf)
        fd = deriv_fd(mf, stepsize) # .reshape(-1, nao, nao)

        for ix in range(natm * 3):
            dv_ref = vmat[ix]
            dv_fd = fd[ix]["dfock"]
            dv_an = an[ix]["dfock"]
            err = abs(dv_ref - dv_an).max()

            print(f"\n{ix = }, stepsize = {stepsize:6.4e}, error = {err:6.4e}")

            print(f"dv_ref = ")
            numpy.savetxt(mf.stdout, dv_ref, fmt='% 8.4e', delimiter=', ')

            print(f"dv_an = ")
            numpy.savetxt(mf.stdout, dv_an, fmt='% 8.4e', delimiter=', ')

        assert 1 == 2


        # for k in fd[ix].keys():
            # dv stands for matrix derivative, du stands for operator derivative
            # if k not in ['ddm_mo', ]:
            #     continue

            # v1 = fd[ix][k] 
            # v2 = an[ix][k]
            # err = abs(v1 - v2).max()

            # v1 = v1[:10, :10]
            # v2 = v2[:10, :10]

            # print(f"\n{k = }, {ix = }, error = {err:6.4e}")
            # print("finite difference:")
            # numpy.savetxt(mf.stdout, v1, fmt='% 8.4f', delimiter=', ')

            # print("analytical:")
            # numpy.savetxt(mf.stdout, v2, fmt='% 8.4f', delimiter=', ')

            # v2 = an[ix]["tov"].T
            # print("\nv2 = ")
            # numpy.savetxt(mf.stdout, v2, fmt='% 8.4e', delimiter=', ')
            # assert 1 == 2


            # print(f"\n{k = }, {ix = }, annalytical = ")
            
            # print(f"{k = }, {ix = }, finite difference = ")
            # numpy.savetxt(mf.stdout, v2, fmt='% 8.4e', delimiter=', ')
            
            # print(f"{k = }, {ix = }, error = {err:6.4e}")
            # assert 1 == 2

            # assert err < 1e-5, "Error too large"