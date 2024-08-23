import os, numpy, scipy, tempfile

import eph.mol
import pyscf
from pyscf import lib, scf, dft
import pyscf.eph
import pyscf.hessian

import eph
from eph.mol import eph_fd, rhf

class ElectronPhononCoupling(eph.mol.rhf.ElectronPhononCoupling):
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

    mf = scf.RKS(mol)
    mf.xc = "PBE0"
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-12
    mf.max_cycle = 1000
    mf.kernel()

    eph_obj = ElectronPhononCoupling(mf)
    eph_obj.grids = None
    dv_sol  = eph_obj.kernel()
    print(dv_sol.shape)

    # Test the finite difference against the analytic results
    # eph_fd = eph.mol.eph_fd.ElectronPhononCoupling(mf)
    # eph_fd.verbose = 0
    # for stepsize in [8e-3, 4e-3, 2e-3, 1e-3, 5e-4]:
    #     dv_ref = eph_fd.kernel(stepsize=stepsize)
    #     err = abs(dv_sol - dv_ref).max()
    #     print("stepsize = % 6.4e, error = % 6.4e" % (stepsize, err))

    from pyscf.eph.eph_fd import gen_moles, run_mfs, get_vmat
    ma, mb = gen_moles(mol, disp=1e-4 / 2.0)
    mfs = run_mfs(mf, ma, mb)
    dv_ref = get_vmat(mf, mfs, disp=1e-4)

    print(dv_sol.shape, dv_ref.shape)

    err = abs(dv_sol - dv_ref).max()
    for ia in range(dv_sol.shape[0]):
        print("\natom %d, error = % 6.4e" % (ia, abs(dv_sol[ia] - dv_ref[ia]).max()))
        print("dv_sol")
        numpy.savetxt(mol.stdout, dv_sol[ia], fmt="% 6.4f", delimiter=", ")

        print("dv_ref")
        numpy.savetxt(mol.stdout, dv_ref[ia], fmt="% 6.4f", delimiter=", ")