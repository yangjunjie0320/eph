import time, pyscf
from pyscf import gto, scf, lib
from eph.mol.eph_fd import ElectronPhononCoupling

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-12
mf.max_cycle = 1000
mf.kernel()

import eph.mol.eph_fd
fd_without_mpi = eph.mol.eph_fd._fd
finalize_without_mpi = eph.mol.eph_fd.ElectronPhononCoupling._finalize

def _fd_with_mpi(**kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ix = kwargs['ix']
    if ix % size == rank:
        res = fd_without_mpi(**kwargs)
        # print(f"ix = {ix:3d}, rank = {rank:3d}")
        return (ix, res)
    else:
        return None
    
eph.mol.eph_fd._fd = _fd_with_mpi

def _finalize_with_mpi(eph_obj, dv_ao):
    dv_ao = comm.gather(dv_ao, root=0)
    if rank == 0:
        dv_ao = filter(lambda x: x is not None, sum(dv_ao, []))
        dv_ao = sorted(dv_ao, key=lambda x: x[0])
        dv_ao = finalize_without_mpi(eph_obj, [x[1] for x in dv_ao])
    
    dv_ao = comm.bcast(dv_ao, root=0)
    return dv_ao

eph.mol.eph_fd.ElectronPhononCoupling._finalize = _finalize_with_mpi

# Test the finite difference against the analytic results
eph_obj = ElectronPhononCoupling(mf)
eph_obj.verbose = 0

t0 = pyscf.lib.logger.timer(mf, "MPI FD")
dv_fd = eph_obj.kernel(stepsize=1e-4)
mf.verbose = 10

if rank == 0:
    pyscf.lib.logger.timer(mf, "MPI(%d) FD" % size, t0)
