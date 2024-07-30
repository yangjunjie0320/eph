import os, sys, pyscf
from pyscf import lib
from pyscf.pbc import gto, scf

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyscf.pbc import gto, scf
cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 0
cell.mesh = [10, 10, 10]
cell.build()

mf = scf.RKS(cell)
mf.xc = "PBE0"
mf.init_guess = 'atom'
mf.verbose = 0
mf.max_cycle = 100
mf.conv_tol = 1e-10
mf.conv_tol_grad = 1e-8
mf.kernel()

import eph.pbc.eph_fd
fd_without_mpi = eph.pbc.eph_fd._fd
finalize_without_mpi = eph.pbc.eph_fd.ElectronPhononCoupling._finalize

def _fd_with_mpi(**kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ix = kwargs['ix']
    if ix % size == rank:
        f = f"fd-ix-{ix}.log"
        f = os.path.join(lib.param.TMPDIR, f)
    
        scf_obj = kwargs['scf_obj']
        scf_obj.verbose = 4
        scf_obj.stdout = open(f, "w")

        res = fd_without_mpi(**kwargs)
        
        scf_obj.stdout.close()
        scf_obj.stdout = sys.stdout
        return (ix, res)
    else:
        return None
    
eph.pbc.eph_fd._fd = _fd_with_mpi

def _finalize_with_mpi(eph_obj, dv_ao):
    dv_ao = comm.gather(dv_ao, root=0)
    if rank == 0:
        dv_ao = filter(lambda x: x is not None, sum(dv_ao, []))
        dv_ao = sorted(dv_ao, key=lambda x: x[0])
        dv_ao = finalize_without_mpi(eph_obj, [x[1] for x in dv_ao])
    
    dv_ao = comm.bcast(dv_ao, root=0)
    return dv_ao

eph.pbc.eph_fd.ElectronPhononCoupling._finalize = _finalize_with_mpi

# Test the finite difference against the analytic results
from eph.pbc.eph_fd import ElectronPhononCoupling
eph_obj = ElectronPhononCoupling(mf)
eph_obj.verbose = 0

t0 = pyscf.lib.logger.timer(mf, "MPI FD")
dv_fd = eph_obj.kernel(stepsize=1e-4)
mf.verbose = 10

if rank == 0:
    pyscf.lib.logger.timer(mf, "MPI(%d) FD" % size, t0)
