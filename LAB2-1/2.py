from mpi4py import MPI
from numpy import empty, array, int32, float64, dot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

if rank == 0:
    f1 = open('in.dat', 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
else:
    N = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)

if rank == 0:
 ave, res = divmod(M, numprocs-1)
 rcounts = empty(numprocs, dtype=int32)
 displs = empty(numprocs, dtype=int32)

 rcounts[0] = 0
displs[0] = 0

for k in range(1, numprocs):
    if k <= res:
        rcounts[k] = ave + 1
    else:
        rcounts[k] = ave
    displs[k] = displs[k-1] + rcounts[k-1]
else: # rank != 0
    rcounts = None
displs = None

M_part = array(0, dtype=int32)
comm.Scatter([rcounts, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

if rank == 0:
    f2 = open('AData.dat', 'r')
    for k in range(1, numprocs):
        A_part = empty((rcounts[k], N), dtype=float64)

for j in range(rcounts[k]):
 for i in range(N):
    A_part[j, i] = float64(f2.readline())

    comm.Send([A_part, rcounts[k]*N, MPI.DOUBLE], dest=k, tag=0)
    f2.close()
    A_part = empty((rcounts[rank], N), dtype=float64)

else: # rank != 0
    A_part = empty((rcounts[rank], N), dtype=float64)
comm.Recv([A_part, rcounts[rank]*N, MPI.DOUBLE],
source=0, status=None)
if rank == 0:
    x = empty(M, dtype=float64)
f3 = open('xData.dat', 'r')
for j in range(M):
    x[j] = float64(f3.readline())
    f3.close()
else:
    x = None

x_part = empty(M_part, dtype=float64)
comm.Scatterv([x, rcounts, displs, MPI.DOUBLE],
[x_part, M_part, MPI.DOUBLE], root=0)
b_temp = dot(A_part.T, x_part)
if rank == 0:
 b = empty(N, dtype=float64)
else:
 b = None
 comm.Reduce([b_temp, N, MPI.DOUBLE],
  [b, N, MPI.DOUBLE], op=MPI.SUM, root=0)
 if rank == 0:
    print(b)