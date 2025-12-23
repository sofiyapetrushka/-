from mpi4py import MPI
import numpy as np

# Инициализация MPI и декартовой топологии
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

# Параметры из лекции
a, b = 0.0, 1.0
t_0, T = 0.0, 2.0
eps = 10**(-1.5)
N, M = 20000, 5000
alpha = 0.5
h = (b - a) / N
tau = (T - t_0) / M

def u_left(t):
    return np.sin(np.pi * t)

def u_right(t):
    return 0.0

# Функции из лекции (без изменений)
def f(y, t, h, N, u_left, u_right, eps):
    f = np.empty(N, dtype=np.float64)
    f[0] = eps * (y[1] - 2 * y[0] + u_left(t)) / h**2 + y[0] * (y[1] - u_left(t)) / (2*h) + y[0]**3
    for n in range(1, N - 1):
        f[n] = eps * (y[n+1] - 2*y[n] + y[n-1]) / h**2 + y[n] * (y[n+1] - y[n-1]) / (2*h) + y[n]**3
    f[N-1] = eps * (u_right(t) - 2*y[N-1] + y[N-2]) / h**2 + y[N-1] * (u_right(t) - y[N-2]) / (2*h) + y[N-1]**3
    return f

def diagonal_preparation(y, t, h, N, u_left, u_right, eps, tau, alpha):
    a = np.empty(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    c = np.empty(N, dtype=np.float64)
    b[0] = 1. - alpha*tau*(-2*eps/h**2 + (y[1] - u_left(t))/(2*h) + 3*y[0]**2)
    c[0] = -alpha * tau * (eps/h**2 + y[0]/(2*h))
    for n in range(1, N - 1):
        a[n] = -alpha*tau*(eps/h**2 - y[n]/(2*h))
        b[n] = 1. - alpha*tau*(-2*eps/h**2 + (y[n+1] - y[n-1])/(2*h) + 3*y[n]**2)
        c[n] = -alpha*tau*(eps/h**2 + y[n]/(2*h))
    a[N-1] = -alpha*tau*(eps/h**2 - y[N-1]/(2*h))
    b[N-1] = 1. - alpha*tau*(-2*eps/h**2 + (u_right(t) - y[N-2])/(2*h) + 3*y[N-1]**2)
    return a, b, c

def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N = len(d)
    x = np.empty(N, dtype=np.float64)
    for n in range(1, N):
        coef = a[n] / b[n-1]
        b[n] = b[n] - coef * c[n-1]
        d[n] = d[n] - coef * d[n-1]
    x[N-1] = d[N-1] / b[N-1]
    for n in range(N-2, -1, -1):
        x[n] = (d[n] - c[n] * x[n+1]) / b[n]
    return x

# Распределение данных
if rank_cart == 0:
    total_points = N + 1
    ave, res = divmod(total_points, numprocs)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    for k in range(numprocs):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
else:
    rcounts, displs = None, None

N_part = np.array(0, dtype=np.int32)
comm_cart.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

# Блоки с гало-ячейками
if rank_cart == 0:
    rcounts_aux = np.empty(numprocs, dtype=np.int32)
    displs_aux = np.empty(numprocs, dtype=np.int32)
    rcounts_aux[0] = rcounts[0] + 1
    displs_aux[0] = 0
    for k in range(1, numprocs - 1):
        rcounts_aux[k] = rcounts[k] + 2
        displs_aux[k] = displs[k] - 1
    if numprocs > 1:
        rcounts_aux[numprocs - 1] = rcounts[numprocs - 1] + 1
        displs_aux[numprocs - 1] = displs[numprocs - 1] - 1
    else:
        rcounts_aux[0] = rcounts[0]
        displs_aux[0] = 0
else:
    rcounts_aux, displs_aux = None, None

N_part_aux = np.array(0, dtype=np.int32)
displs_aux_val = np.array(0, dtype=np.int32)
comm_cart.Scatter([rcounts_aux, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)
comm_cart.Scatter([displs_aux, 1, MPI.INT], [displs_aux_val, 1, MPI.INT], root=0)

# Инициализация решения
if rank_cart == 0:
    x_global = np.linspace(a, b, N + 1)
    u_global = np.sin(np.pi * x_global)
else:
    u_global = None

u_part = np.empty(N_part, dtype=np.float64)
comm_cart.Scatterv([u_global, rcounts, displs, MPI.DOUBLE], u_part, root=0)

u_part_aux = np.empty(N_part_aux, dtype=np.float64)
if N_part_aux == N_part:
    u_part_aux[:] = u_part
else:
    u_part_aux[1:-1] = u_part
    # Инициализация гало
    if rank_cart == 0:
        u_part_aux[0] = u_left(0.0)
    if rank_cart == numprocs - 1:
        u_part_aux[-1] = u_right(0.0)

y_part = u_part.copy()

# Временной цикл
for m in range(M):
    t = t_0 + m * tau

    # Обновление гало в u_part_aux
    if N_part_aux != N_part:
        if rank_cart == 0:
            comm_cart.Sendrecv(
                sendbuf=[y_part[-1:], 1, MPI.DOUBLE], dest=1, sendtag=0,
                recvbuf=[u_part_aux[-1:], 1, MPI.DOUBLE], source=1, recvtag=1
            )
            u_part_aux[0] = u_left(t)
            u_part_aux[1:-1] = y_part
        elif rank_cart == numprocs - 1:
            comm_cart.Sendrecv(
                sendbuf=[y_part[:1], 1, MPI.DOUBLE], dest=numprocs - 2, sendtag=1,
                recvbuf=[u_part_aux[:1], 1, MPI.DOUBLE], source=numprocs - 2, recvtag=0
            )
            u_part_aux[-1] = u_right(t)
            u_part_aux[1:-1] = y_part
        else:
            comm_cart.Sendrecv(
                sendbuf=[y_part[:1], 1, MPI.DOUBLE], dest=rank_cart - 1, sendtag=1,
                recvbuf=[u_part_aux[:1], 1, MPI.DOUBLE], source=rank_cart - 1, recvtag=0
            )
            comm_cart.Sendrecv(
                sendbuf=[y_part[-1:], 1, MPI.DOUBLE], dest=rank_cart + 1, sendtag=0,
                recvbuf=[u_part_aux[-1:], 1, MPI.DOUBLE], source=rank_cart + 1, recvtag=1
            )
            u_part_aux[1:-1] = y_part

    # ROS1: вычисление правой части и матрицы
    a_vec, b_vec, c_vec = diagonal_preparation(
        u_part_aux, t, h, N_part_aux, u_left, u_right, eps, tau, alpha
    )
    d_vec = f(u_part_aux, t, h, N_part_aux, u_left, u_right, eps)

    # Локальное решение СЛАУ (как в лекции — "параллельная" прогонка = локальная)
    w_1_part = consecutive_tridiagonal_matrix_algorithm(a_vec, b_vec, c_vec, d_vec)

    # Обновление решения
    if N_part_aux == N_part:
        y_part += tau * w_1_part
    else:
        y_part += tau * w_1_part[1:-1]

# Сбор результата
u_solution = np.empty(N + 1, dtype=np.float64) if rank_cart == 0 else None
comm_cart.Gatherv([y_part, N_part, MPI.DOUBLE], [u_solution, rcounts, displs, MPI.DOUBLE], root=0)

if rank_cart == 0:
    u_solution[0] = u_left(T)
    u_solution[-1] = u_right(T)
    np.save("solution.npy", u_solution)

comm_cart.Free()