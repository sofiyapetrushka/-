from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot, zeros_like
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def conjugate_gradient_method(A_part, b_part, x_part, 
                              N, N_part, rcounts_N, displs_N):
    
    # Для случая с одним процессом используем последовательную версию
    if numprocs == 1:
        return sequential_conjugate_gradient(A_part, b_part, x_part)
    
    x = empty(N, dtype=float64)
    p = empty(N, dtype=float64)
    
    r_part = empty(N_part, dtype=float64)
    p_part = zeros(N_part, dtype=float64)  # Инициализируем нулями
    q_part = empty(N_part, dtype=float64)
    
    ScalP = array(0, dtype=float64)
    ScalP_temp = empty(1, dtype=float64)
    
    s = 1

    while s <= N:

        if s == 1:
            # Собираем полный вектор x
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE], 
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            
            # Вычисляем r_temp = A_part^T * (A_part * x - b_part)
            Ax_minus_b = dot(A_part, x) - b_part
            r_temp = dot(A_part.T, Ax_minus_b)
            
            # Распределяем r_temp по процессам
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE], 
                                [r_part, N_part, MPI.DOUBLE], 
                                recvcounts=rcounts_N, op=MPI.SUM)
        else:
            ScalP_temp[0] = dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                           [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            if ScalP != 0:  # Защита от деления на ноль
                r_part = r_part - q_part/ScalP
            
        # Обновление p_part
        ScalP_temp[0] = dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        if ScalP != 0:  # Защита от деления на ноль
            p_part = p_part + r_part/ScalP
           
        # Собираем полный вектор p
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        
        # Вычисляем q_temp = A_part^T * (A_part * p)
        Ap = dot(A_part, p)
        q_temp = dot(A_part.T, Ap)
        
        # Распределяем q_temp по процессам
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE], 
                            recvcounts=rcounts_N, op=MPI.SUM)
        
        # Обновление x_part
        ScalP_temp[0] = dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        if ScalP != 0:  # Защита от деления на ноль
            x_part = x_part - p_part/ScalP
        
        s = s + 1
    
    return x_part

def sequential_conjugate_gradient(A, b, x):
    """Последовательная версия метода сопряженных градиентов для одного процесса"""
    N = x.shape[0]
    r = dot(A.T, dot(A, x) - b)
    p = zeros_like(r)
    
    for s in range(1, N + 1):
        if s == 1:
            r_current = r.copy()
        else:
            scal_p = dot(p, q)
            if scal_p != 0:
                r_current = r_current - q / scal_p
            
        scal_r = dot(r_current, r_current)
        if scal_r != 0:
            p = p + r_current / scal_r
        
        q = dot(A.T, dot(A, p))
        
        scal_pq = dot(p, q)
        if scal_pq != 0:
            x = x - p / scal_pq
    
    return x

def auxiliary_arrays_determination(M, numprocs):
    """Определение вспомогательных массивов с проверкой на 1 процесс"""
    if numprocs == 1:
        # Для одного процесса все данные на процессе 0
        rcounts = array([M], dtype=int32)
        displs = array([0], dtype=int32)
    else:
        # Для нескольких процессов распределяем данные
        ave, res = divmod(M, numprocs)
        rcounts = empty(numprocs, dtype=int32)
        displs = empty(numprocs, dtype=int32)
        
        displs[0] = 0
        for k in range(numprocs):
            if k < res:
                rcounts[k] = ave + 1
            else:
                rcounts[k] = ave
            if k > 0:
                displs[k] = displs[k-1] + rcounts[k-1]
    
    return rcounts, displs

# Основная программа
if rank == 0:
    try:
        f1 = open('in.dat', 'r')
        N = array(int32(f1.readline()))
        M = array(int32(f1.readline()))
        f1.close()
        print(f"N = {N}, M = {M}")
    except Exception as e:
        print(f"Error reading in.dat: {e}")
        N = array(0, dtype=int32)
        M = array(0, dtype=int32)
else:
    N = array(0, dtype=int32)
    M = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)
comm.Bcast([M, 1, MPI.INT], root=0)

# Проверяем, что N и M корректны
if N == 0 or M == 0:
    if rank == 0:
        print("Error: N or M is zero")
    MPI.Finalize()
    exit()

# Определяем массивы распределения
if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)
    print(f"rcounts_M: {rcounts_M}, displs_M: {displs_M}")
    print(f"rcounts_N: {rcounts_N}, displs_N: {displs_N}")
else:
    rcounts_M = empty(numprocs, dtype=int32)
    displs_M = empty(numprocs, dtype=int32)
    rcounts_N = empty(numprocs, dtype=int32)
    displs_N = empty(numprocs, dtype=int32)

comm.Bcast([rcounts_M, numprocs, MPI.INT], root=0)
comm.Bcast([displs_M, numprocs, MPI.INT], root=0)
comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
comm.Bcast([displs_N, numprocs, MPI.INT], root=0)

M_part = rcounts_M[rank]
N_part = rcounts_N[rank]

print(f"Rank {rank}: M_part = {M_part}, N_part = {N_part}")

# Загрузка и распределение матрицы A
A_part = empty((M_part, N), dtype=float64)

if rank == 0:
    try:
        f2 = open('AData.dat', 'r')
        # Процесс 0 читает свою часть
        for j in range(M_part):
            for i in range(N):
                A_part[j, i] = float64(f2.readline())
        
        # Отправляем части другим процессам
        for k in range(1, numprocs):
            if rcounts_M[k] > 0:
                A_temp = empty((rcounts_M[k], N), dtype=float64)
                for j in range(rcounts_M[k]):
                    for i in range(N):
                        A_temp[j, i] = float64(f2.readline())
                comm.Send([A_temp, rcounts_M[k] * N, MPI.DOUBLE], dest=k, tag=0)
        f2.close()
    except Exception as e:
        print(f"Error reading AData.dat: {e}")
else:
    if M_part > 0:
        comm.Recv([A_part, M_part * N, MPI.DOUBLE], source=0, tag=0)

# Загрузка и распределение вектора b
if rank == 0:
    try:
        b = empty(M, dtype=float64)
        f3 = open('bData.dat', 'r')
        for j in range(M):
            b[j] = float64(f3.readline())
        f3.close()
    except Exception as e:
        print(f"Error reading bData.dat: {e}")
        b = zeros(M, dtype=float64)
else:
    b = None

b_part = empty(M_part, dtype=float64)

if numprocs > 1:
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
                  [b_part, M_part, MPI.DOUBLE], root=0)
else:
    if rank == 0:
        b_part = b.copy()

# Инициализация и распределение вектора x
if rank == 0:
    x = zeros(N, dtype=float64)
else:
    x = None

x_part = empty(N_part, dtype=float64)

if numprocs > 1:
    comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], 
                  [x_part, N_part, MPI.DOUBLE], root=0)
else:
    if rank == 0:
        x_part = x.copy()

# Запуск метода сопряженных градиентов
if M_part > 0 and N_part > 0:
    x_part = conjugate_gradient_method(A_part, b_part, x_part, 
                                       N, N_part, rcounts_N, displs_N)
else:
    # Если у процесса нет данных, создаем пустой массив
    x_part = zeros(N_part, dtype=float64)

# Сбор результатов
if rank == 0:
    x_result = empty(N, dtype=float64)
else:
    x_result = None

comm.Gatherv([x_part, N_part, MPI.DOUBLE], 
             [x_result, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x_result, '-y', lw=3)
    show()