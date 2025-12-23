from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Шаг 1. Процесс 0 читает размеры матрицы
if rank == 0:
    with open('matr.dat', 'r') as f:
        M = int(f.readline())
        N = int(f.readline())
else:
    M = None
    N = None

# Рассылаем размеры всем процессам
M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)

# Процесс 0 читает всю матрицу и вектор
if rank == 0:
    A_full = np.loadtxt('amatr.dat').reshape(M, N)
    x = np.loadtxt('vektor.dat')
else:
    A_full = None
    x = np.empty(N, dtype='d')  # подготовка для рассылки

# Расчет размеров блоков для Scatterv
# Определяем размеры и смещения
counts = np.ones(size, dtype=int) * (M // size)
remainder = M % size
for i in range(remainder):
    counts[i] += 1

displs = np.zeros(size, dtype=int)
displs[0] = 0
for i in range(1, size):
    displs[i] = displs[i - 1] + counts[i - 1]

# Подготовка буфера для локальной части матрицы
local_M = counts[rank]
A_local = np.empty((local_M, N), dtype='d')

# Рассылка части матрицы с помощью Scatterv
comm.Scatterv([A_full, counts * N, displs * N, MPI.DOUBLE], A_local, root=0)

# Рассылка вектора `x`
comm.Bcast(x, root=0)

# Локальное умножение
b_local = np.dot(A_local, x)

# Подготовка для сборки результата через Gatherv
recv_counts = counts
recv_displs = displs

if rank == 0:
    b = np.empty(M, dtype='d')
else:
    b = None

# Собираем локальные результаты через Gatherv
comm.Gatherv(b_local, [b, recv_counts, recv_displs, MPI.DOUBLE], root=0)

# Процесс 0 сохраняет результат
if rank == 0:
    np.savetxt('results_parallel2.dat', b)