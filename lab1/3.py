from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    with open('matr.dat', 'r') as f:
        M = int(f.readline())
        N = int(f.readline())
else:
    M = None
    N = None

M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)

local_M = M // size  
A_local = np.empty((local_M, N), dtype='d') 

if rank == 0:
    A = np.loadtxt('amatr.dat').reshape(M, N)
    x = np.loadtxt('vektor.dat')
else:
    A = None
    x = np.empty(N, dtype='d')

comm.Scatter([A, MPI.DOUBLE], [A_local, MPI.DOUBLE], root=0)

comm.Bcast(x, root=0)

b_local = np.dot(A_local, x)

if rank == 0:
    b = np.empty(M, dtype='d')
else:
    b = None

comm.Gather([b_local, MPI.DOUBLE], [b, MPI.DOUBLE], root=0)

if rank == 0:
    np.savetxt('results_parallel.dat', b)