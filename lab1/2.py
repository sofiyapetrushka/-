from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    
    with open('matr.dat', 'r') as f:
        M = int(f.readline())
        N = int(f.readline())


M = comm.bcast(M if rank == 0 else None, root=0)
N = comm.bcast(N if rank == 0 else None, root=0)


if rank == 0:
    A = np.loadtxt('amatr.dat').reshape(M, N)
    x = np.loadtxt('vektor.dat')
else:
    A = None
    x = np.empty(N, dtype=float)


local_M = M // size
A_local = np.empty((local_M, N), dtype=float)


for i in range(size):
    start_row = i * local_M
    end_row = start_row + local_M
    if rank == i:
        
        if rank != 0:
            comm.Send(A[start_row:end_row, :], dest=0)
        else:
            A_local = A[start_row:end_row, :]
    elif rank != 0:
        
        comm.Recv(A_local, source=i)


comm.Bcast(x, root=0)


b_local = np.dot(A_local, x)


b = None
if rank == 0:
    b = np.empty(M, dtype=float)

for i in range(size):
    start_row = i * local_M
    end_row = start_row + local_M
    if rank == 0:
        b[start_row:end_row] = b_local if i == 0 else None
    if i != 0:
        comm.Recv(b[start_row:end_row], source=i)
    elif rank == 0:
        b[start_row:end_row] = b_local


if rank == 0:
    np.savetxt('res_parallel.dat', b)