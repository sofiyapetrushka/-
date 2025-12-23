from mpi4py import MPI 
import numpy as np
import time

# Инициализация MPI
comm = MPI.COMM_WORLD 
numprocs = comm.Get_size() 
rank = comm.Get_rank() 

print(f"Процесс {rank}: старт, всего процессов = {numprocs}")

# Проверка количества процессов
if numprocs < 2:
    if rank == 0:
        print("ОШИБКА: Нужно минимум 2 процесса!")
        print("Запустите: mpiexec -n 4 python script.py")
    MPI.Finalize()
    exit()

# Создание данных для отправки и приема
a = np.array([rank], dtype=np.int32)          # Что отправляем (текущий ранг)
a_recv = np.empty(1, dtype=np.int32)          # Куда принимаем

# Определение соседей в кольце
next_proc = (rank + 1) % numprocs             # Следующий процесс
prev_proc = (rank - 1) % numprocs             # Предыдущий процесс

print(f"Процесс {rank}: отправляю {next_proc}, получаю от {prev_proc}")

# СОЗДАНИЕ ОТЛОЖЕННЫХ (persistent) ЗАПРОСОВ
# Send_init - создает отложенный запрос на отправку
# Recv_init - создает отложенный запрос на прием
requests = [MPI.REQUEST_NULL for _ in range(2)]
requests[0] = comm.Send_init([a, MPI.INT], dest=next_proc, tag=0)
requests[1] = comm.Recv_init([a_recv, MPI.INT], source=prev_proc, tag=0)

# Начало измерения времени
start_time = time.time()

# ЦИКЛ ИЗ 10 ИТЕРАЦИЙ
for iteration in range(10):
    print(f"Процесс {rank}: итерация {iteration + 1}, отправляю значение {a[0]}")
    
    # АКТИВАЦИЯ отложенных запросов для текущей итерации
    MPI.Prequest.Startall(requests)
    
    # ОЖИДАНИЕ завершения операций текущей итерации
    MPI.Request.Waitall(requests)
    
    # ОБНОВЛЕНИЕ данных для следующей итерации
    # Принятое значение становится отправляемым
    a[0] = a_recv[0]
    
    print(f"Процесс {rank}: итерация {iteration + 1} завершена, получил {a_recv[0]}")

# Освобождение ресурсов отложенных запросов
for req in requests:
    req.Free()

# Конец измерения времени
end_time = time.time()

print(f"Процесс {rank}: завершил за {end_time - start_time:.6f} секунд")
print(f"Процесс {rank}: финальное значение = {a[0]}")
print("-" * 50)