from mpi4py import MPI 
import numpy as np
import time

comm = MPI.COMM_WORLD 
numprocs = comm.Get_size() 
rank = comm.Get_rank() 

print(f"Процесс {rank}: работа с матрицами")

if numprocs < 2:
    if rank == 0:
        print("Нужно минимум 2 процесса!")
    exit()

# Размер матрицы (3x3)
ROWS, COLS = 3, 3

# Создание матриц
# Исходная матрица заполняется номером процесса
matrix_send = np.full((ROWS, COLS), rank, dtype=np.int32)
matrix_recv = np.empty((ROWS, COLS), dtype=np.int32)

print(f"Процесс {rank}: исходная матрица:\n{matrix_send}")

# Определение соседей
next_proc = (rank + 1) % numprocs
prev_proc = (rank - 1) % numprocs

# СОЗДАНИЕ ОТЛОЖЕННЫХ ЗАПРОСОВ для матриц
requests = [MPI.REQUEST_NULL for _ in range(2)]
requests[0] = comm.Send_init([matrix_send, MPI.INT], dest=next_proc, tag=0)
requests[1] = comm.Recv_init([matrix_recv, MPI.INT], source=prev_proc, tag=0)

start_time = time.time()

# ЦИКЛ ОБМЕНА
ITERATIONS = 5
for i in range(ITERATIONS):
    print(f"\nПроцесс {rank}: итерация {i+1}")
    print(f"Отправляю матрицу (все элементы = {matrix_send[0,0]}):")
    print(matrix_send)
    
    # Запуск операций
    MPI.Prequest.Startall(requests)
    MPI.Request.Waitall(requests)
    
    # Обновление: принятая матрица становится отправляемой
    matrix_send[:] = matrix_recv[:]  # Копируем все элементы
    
    print(f"Получил матрицу (все элементы = {matrix_recv[0,0]}):")
    print(matrix_recv)

# Освобождение ресурсов
for req in requests:
    req.Free()

end_time = time.time()

print(f"\nПроцесс {rank}: время выполнения = {end_time - start_time:.6f} сек")
print(f"Процесс {rank}: финальная матрица (после {ITERATIONS} итераций):")
print(matrix_send)
print("=" * 60)