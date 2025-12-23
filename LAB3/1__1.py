from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot
from matplotlib.pyplot import style, figure, axes, show
import numpy as np

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def conjugate_gradient_method(A_part, b_part, x_part, N, N_part, rcounts_N, displs_N, rcounts_M, displs_M, M):
    """
    Параллельная реализация метода сопряжённых градиентов
    """
    # Векторы, которые должны быть полными на всех процессах
    x_full = empty(N, dtype=float64)
    p_full = empty(N, dtype=float64)
    
    # Локальные векторы
    r_part = zeros(N_part, dtype=float64)
    p_part = zeros(N_part, dtype=float64)
    q_part = zeros(N_part, dtype=float64)
    
    # Скалярные произведения (используем массивы вместо скаляров)
    rho_prev = array([0.0], dtype=float64)
    rho_curr = array([0.0], dtype=float64)
    alpha = array([0.0], dtype=float64)
    beta = array([0.0], dtype=float64)
    p_dot_q = array([0.0], dtype=float64)
    
    # Инициализация
    comm.Allgatherv([x_part, N_part, MPI.DOUBLE], 
                    [x_full, rcounts_N, displs_N, MPI.DOUBLE])
    
    # Вычисление начальной невязки: r = A^T * (A * x - b)
    Ax_part = dot(A_part, x_full)  # A_part * x_full
    Ax_minus_b_part = Ax_part - b_part  # (A*x - b) для локальной части
    
    # Собираем полный вектор Ax_minus_b для всех процессов
    Ax_minus_b_full = empty(M, dtype=float64)
    comm.Allgatherv([Ax_minus_b_part, M_part, MPI.DOUBLE], 
                    [Ax_minus_b_full, rcounts_M, displs_M, MPI.DOUBLE])
    
    # Вычисляем r = A^T * (A*x - b)
    r_temp = dot(A_part.T, Ax_minus_b_full)
    comm.Reduce_scatter([r_temp, N, MPI.DOUBLE], 
                        [r_part, N_part, MPI.DOUBLE], 
                        recvcounts=rcounts_N, op=MPI.SUM)
    
    p_part = r_part.copy()
    
    # Начальное скалярное произведение (r, r)
    rho_curr[0] = dot(r_part, r_part)
    temp_rho = array([rho_curr[0]], dtype=float64)
    comm.Allreduce([temp_rho, MPI.DOUBLE], [rho_curr, MPI.DOUBLE], op=MPI.SUM)
    
    max_iter = N
    tolerance = 1e-12
    
    for iteration in range(max_iter):
        # Проверка сходимости
        if rho_curr[0] < tolerance:
            if rank == 0:
                print(f"Сходимость достигнута на итерации {iteration}")
            break
        
        # Собираем полный вектор p
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p_full, rcounts_N, displs_N, MPI.DOUBLE])
        
        # Вычисляем q = A^T * (A * p)
        Ap_part = dot(A_part, p_full)  # A_part * p_full
        
        # Собираем полный вектор Ap
        Ap_full = empty(M, dtype=float64)
        comm.Allgatherv([Ap_part, M_part, MPI.DOUBLE], 
                        [Ap_full, rcounts_M, displs_M, MPI.DOUBLE])
        
        # Вычисляем q = A^T * Ap
        q_temp = dot(A_part.T, Ap_full)
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE], 
                            recvcounts=rcounts_N, op=MPI.SUM)
        
        # Вычисляем alpha = rho_curr / (p, q)
        p_dot_q[0] = dot(p_part, q_part)
        temp_p_dot_q = array([p_dot_q[0]], dtype=float64)
        comm.Allreduce([temp_p_dot_q, MPI.DOUBLE], [p_dot_q, MPI.DOUBLE], op=MPI.SUM)
        
        alpha[0] = rho_curr[0] / p_dot_q[0]
        
        # Обновляем решение и невязку
        x_part += alpha[0] * p_part
        r_part -= alpha[0] * q_part
        
        # Сохраняем предыдущее значение rho
        rho_prev[0] = rho_curr[0]
        
        # Вычисляем новое значение rho
        rho_curr[0] = dot(r_part, r_part)
        temp_rho = array([rho_curr[0]], dtype=float64)
        comm.Allreduce([temp_rho, MPI.DOUBLE], [rho_curr, MPI.DOUBLE], op=MPI.SUM)
        
        # Вычисляем beta для следующей итерации
        beta[0] = rho_curr[0] / rho_prev[0]
        
        # Обновляем направление
        p_part = r_part + beta[0] * p_part
        
        if rank == 0 and iteration % 100 == 0:
            print(f"Итерация {iteration}, невязка: {rho_curr[0]:.6e}")
    
    if rank == 0:
        print(f"Метод завершен за {iteration} итераций, конечная невязка: {rho_curr[0]:.6e}")
    
    return x_part

def auxiliary_arrays_determination(total_size, numprocs):
    """Функция для расчета массивов rcounts и displs"""
    ave, res = divmod(total_size, numprocs)
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
    # Чтение параметров из файла
    print("Чтение параметров из файла in.dat...")
    with open('in.dat', 'r') as f1:
        N = array(int32(f1.readline()))
        M = array(int32(f1.readline()))
    print(f"Размерность задачи: N={N}, M={M}")
else:
    N = array(0, dtype=int32)
    M = array(0, dtype=int32)

# Рассылаем размерности всем процессам
comm.Bcast([N, 1, MPI.INT], root=0)
comm.Bcast([M, 1, MPI.INT], root=0)

# Вычисляем массивы для распределения данных
if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)
    print("Массивы распределения вычислены")
else:
    rcounts_M = empty(numprocs, dtype=int32)
    displs_M = empty(numprocs, dtype=int32)
    rcounts_N = empty(numprocs, dtype=int32)
    displs_N = empty(numprocs, dtype=int32)

# Рассылаем массивы распределения
comm.Bcast([rcounts_M, numprocs, MPI.INT], root=0)
comm.Bcast([displs_M, numprocs, MPI.INT], root=0)
comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
comm.Bcast([displs_N, numprocs, MPI.INT], root=0)

# Определяем локальные размеры
M_part = rcounts_M[rank]
N_part = rcounts_N[rank]

if rank == 0:
    print(f"Распределение: M_part = {rcounts_M}, N_part = {rcounts_N}")

# Распределение матрицы A
if rank == 0:
    print("Чтение матрицы A из файла AData.dat...")
    A_part = empty((M_part, N), dtype=float64)
    with open('AData.dat', 'r') as f2:
        # Читаем свою часть матрицы A
        for j in range(M_part):
            for i in range(N):
                A_part[j, i] = float64(f2.readline())
    
    # Отправляем части другим процессам
    for k in range(1, numprocs):
        A_temp = empty((rcounts_M[k], N), dtype=float64)
        for j in range(rcounts_M[k]):
            for i in range(N):
                A_temp[j, i] = float64(f2.readline())
        comm.Send([A_temp, rcounts_M[k] * N, MPI.DOUBLE], dest=k, tag=0)
    print("Матрица A успешно распределена")
else:
    A_part = empty((M_part, N), dtype=float64)
    comm.Recv([A_part, M_part * N, MPI.DOUBLE], source=0, tag=0)

# Распределение вектора b
if rank == 0:
    print("Чтение вектора b из файла bData.dat...")
    b = empty(M, dtype=float64)
    with open('bData.dat', 'r') as f3:
        for j in range(M):
            b[j] = float64(f3.readline())
    print("Вектор b успешно прочитан")
else:
    b = None

b_part = empty(M_part, dtype=float64)
comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
              [b_part, M_part, MPI.DOUBLE], root=0)

# Инициализация начального приближения x
if rank == 0:
    x = zeros(N, dtype=float64)
    print("Начальное приближение x инициализировано нулями")
else:
    x = None

x_part = empty(N_part, dtype=float64)
comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], 
              [x_part, N_part, MPI.DOUBLE], root=0)

# Вызов метода сопряжённых градиентов
if rank == 0:
    print("Запуск метода сопряжённых градиентов...")

x_part = conjugate_gradient_method(A_part, b_part, x_part, N, N_part, rcounts_N, displs_N, rcounts_M, displs_M, M)

# Сбор результатов
comm.Gatherv([x_part, N_part, MPI.DOUBLE], 
             [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

# Вывод результата на процессе 0
if rank == 0:
    print("\nРешение найдено!")
    
    # Вывод первых и последних элементов решения
    print("Первые 10 элементов решения:")
    for i in range(min(10, N)):
        print(f"x[{i}] = {x[i]:.6e}")
    
    if N > 10:
        print("Последние 10 элементов решения:")
        for i in range(max(0, N-10), N):
            print(f"x[{i}] = {x[i]:.6e}")
    
    # Визуализация результата
    style.use('default')
    fig = figure(figsize=(12, 6))
    ax = axes()
    ax.set_xlabel('Индекс i')
    ax.set_ylabel('x[i]')
    ax.set_title(f'Решение системы уравнений (N={N})')
    ax.plot(arange(N), x, 'b-', lw=1.5, marker='', markersize=1)
    ax.grid(True, alpha=0.3)
    
    # Сохранение результата в файл
    with open('solution.dat', 'w') as f:
        for i in range(N):
            f.write(f"{x[i]}\n")
    print(f"\nРешение сохранено в файл solution.dat")
    
    show()