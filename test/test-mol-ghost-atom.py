import os, sys
import numpy, scipy

import pyscf
from pyscf import gto, scf, ao2mo

import eph

def deriv_fd(mf, stepsize=1e-4):
    natm = mf.mol.natm
    symbs = [mf.mol.atom_pure_symbol(ia) for ia in range(natm)]
    x0 = mf.mol.atom_coords(unit='Bohr')

    basis = mf.mol.basis
    nao = mf.mol.nao_nr()
    
    res = []
    for ix in range(natm * 3):
        ia, x = divmod(ix, 3)

        dx = numpy.zeros_like(x0)
        dx[ia, x] += stepsize
        
        atom  = [("X:" + s,   x0[ia]) for ia, s in enumerate(symbs)]
        atom += [(s, x0[ia] + dx[ia]) for ia, s in enumerate(symbs)]

        m1 = gto.M(
            verbose=0, unit = "Bohr",
            atom = atom, basis = basis
        )
        int1e_nuc = m1.intor("int1e_nuc")
        assert int1e_nuc.shape == (2 * nao, 2 * nao)

        v1 = int1e_nuc[:nao, :nao] # in
        u1 = int1e_nuc[nao:, nao:]

        atom  = [("X:" + s,   x0[ia]) for ia, s in enumerate(symbs)]
        atom += [(s, x0[ia] - dx[ia]) for ia, s in enumerate(symbs)]

        m2 = gto.M(
            verbose=0, unit = "Bohr",
            atom = atom, basis = basis
        )
        int1e_nuc = m2.intor("int1e_nuc")
        assert int1e_nuc.shape == (2 * nao, 2 * nao)

        v2 = int1e_nuc[:nao, :nao] # in
        u2 = int1e_nuc[nao:, nao:]

        dv = (v1 - v2) / (2 * stepsize)
        du = (u1 - u2) / (2 * stepsize)

        res.append(
            {
                'dv': dv,
                'du': du,
            }
        )

    return res

def deriv_an(mf, stepsize=1e-4):
    pass


if __name__ == '__main__':
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