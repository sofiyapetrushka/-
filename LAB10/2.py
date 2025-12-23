from mpi4py import MPI
import numpy as np
import time

# Инициализация MPI
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

a, b = -2.0, 2.0
c, d = -2.0, 2.0
t_0, T = 0.0, 4.0
eps = 10**(-1.5)
N_x, N_y = 200, 200
M = 4000

h_x = (b - a) / N_x
h_y = (d - c) / N_y
tau = T / M

#Граничные и начальные условия
def u_init(x, y):
    return 0.5 * np.tanh(1 / eps * ((x - 0.5)**2 + (y - 0.5)**2 - 0.35**2)) - 0.17

def u_left(y, t):    return 0.33
def u_right(y, t):   return 0.33
def u_bottom(x, t):  return 0.33
def u_top(x, t):     return 0.33

# Вспомогательная функция для распределения
def auxiliary_arrays_determination(M, num):
    ave, res = divmod(M, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    for k in range(num):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
    return rcounts, displs

# Распределение по X
total_x = N_x + 1
rcounts_N_x, displs_N_x = auxiliary_arrays_determination(total_x, numprocs)
N_x_part = rcounts_N_x[rank_cart]

# Размер с гало-слоями
if numprocs == 1:
    N_x_part_aux = N_x_part
    displs_N_x_aux = displs_N_x.copy()
else:
    if rank_cart == 0 or rank_cart == numprocs - 1:
        N_x_part_aux = N_x_part + 1
    else:
        N_x_part_aux = N_x_part + 2
    # Смещения для блоков с гало
    displs_N_x_aux = displs_N_x - 1
    displs_N_x_aux[0] = 0

displ_x_aux = displs_N_x_aux[rank_cart]

# Инициализация локального блока
u_part_aux = np.empty((M + 1, N_x_part_aux, N_y + 1), dtype=np.float64)

# Глобальная инициализация (только на rank 0)
if rank_cart == 0:
    x_global = np.linspace(a, b, total_x)
    y_global = np.linspace(c, d, N_y + 1)
    u_global = np.empty((M + 1, total_x, N_y + 1), dtype=np.float64)
    for i in range(total_x):
        for j in range(N_y + 1):
            u_global[0, i, j] = u_init(x_global[i], y_global[j])
    # Граничные условия на t=0
    for j in range(N_y + 1):
        u_global[0, 0, j] = u_left(y_global[j], t_0)
        u_global[0, N_x, j] = u_right(y_global[j], t_0)
    for i in range(total_x):
        u_global[0, i, 0] = u_bottom(x_global[i], t_0)
        u_global[0, i, N_y] = u_top(x_global[i], t_0)
else:
    u_global = None
    x_global = y_global = None

# Рассылаем только начальный слой (m=0)
u0_slice = None
if rank_cart == 0:
    u0_slice = u_global[0]

u0_local = np.empty((N_x_part, N_y + 1), dtype=np.float64)
comm_cart.Scatterv(
    [u0_slice, rcounts_N_x * (N_y + 1), displs_N_x * (N_y + 1), MPI.DOUBLE],
    u0_local,
    root=0
)

# Заполняем u_part_aux[0]
if numprocs == 1:
    u_part_aux[0] = u0_local
else:
    u_part_aux[0, 1:-1, :] = u0_local
    # Установим граничные значения позже — после обмена

# Основной цикл по времени
start_time = time.time() if rank_cart == 0 else None

for m in range(M):
    t_current = t_0 + m * tau

    # Установка граничных условий по Y (всегда)
    for i_local in range(N_x_part_aux):
        u_part_aux[m + 1, i_local, 0] = u_bottom(a + (displ_x_aux + i_local) * h_x, t_current + tau)
        u_part_aux[m + 1, i_local, N_y] = u_top(a + (displ_x_aux + i_local) * h_x, t_current + tau)

    # Установка граничных условий по X (глобальные границы)
    if rank_cart == 0:
        for j in range(N_y + 1):
            u_part_aux[m + 1, 0, j] = u_left(c + j * h_y, t_current + tau)
    if rank_cart == numprocs - 1:
        for j in range(N_y + 1):
            u_part_aux[m + 1, -1, j] = u_right(c + j * h_y, t_current + tau)

    # Вычисления во внутренних узлах (включая "внутренние границы" между процессами)
    i_start = 1 if rank_cart == 0 else 1
    i_end = N_x_part_aux - 1 if rank_cart == numprocs - 1 else N_x_part_aux - 1

    for i in range(i_start, i_end):
        for j in range(1, N_y):
            d2x = (u_part_aux[m, i+1, j] - 2*u_part_aux[m, i, j] + u_part_aux[m, i-1, j]) / h_x**2
            d2y = (u_part_aux[m, i, j+1] - 2*u_part_aux[m, i, j] + u_part_aux[m, i, j-1]) / h_y**2
            d1x = (u_part_aux[m, i+1, j] - u_part_aux[m, i-1, j]) / (2*h_x)
            d1y = (u_part_aux[m, i, j+1] - u_part_aux[m, i, j-1]) / (2*h_y)
            u_part_aux[m+1, i, j] = u_part_aux[m, i, j] + tau * (
                eps * (d2x + d2y) +
                u_part_aux[m, i, j] * (d1x + d1y) +
                u_part_aux[m, i, j]**3
            )

    # Обмен гало-слоями
    if numprocs > 1:
        if rank_cart > 0:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m+1, 1, :], N_y+1, MPI.DOUBLE],
                dest=rank_cart-1, sendtag=0,
                recvbuf=[u_part_aux[m+1, 0, :], N_y+1, MPI.DOUBLE],
                source=rank_cart-1, recvtag=MPI.ANY_TAG
            )
        if rank_cart < numprocs - 1:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m+1, -2, :], N_y+1, MPI.DOUBLE],
                dest=rank_cart+1, sendtag=0,
                recvbuf=[u_part_aux[m+1, -1, :], N_y+1, MPI.DOUBLE],
                source=rank_cart+1, recvtag=MPI.ANY_TAG
            )

# Сбор результатов (опционально, только для небольших задач)
if rank_cart == 0:
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.4f} sec")

np.savez(f"results_part_{rank_cart}.npz",
         u=u_part_aux,
         rank=rank_cart,
         displ_x=displ_x_aux,
         N_x_part=N_x_part_aux,
         N_y=N_y+1)

comm_cart.Free()