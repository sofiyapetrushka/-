from mpi4py import MPI 
import numpy as np
import time

comm = MPI.COMM_WORLD 
numprocs = comm.Get_size() 
rank = comm.Get_rank() 

if numprocs < 2:
    if rank == 0:
        print("Запустите с 2+ процессами!")
    exit()

# Определение соседей
next_proc = (rank + 1) % numprocs
prev_proc = (rank - 1) % numprocs

# Тестовые данные
data_size = 1000
iterations = 100

print(f"\nПроцесс {rank}: сравнение производительности")
print(f"Размер данных: {data_size} элементов, итераций: {iterations}")

# ========== ТЕСТ 1: Отложенные запросы (Persistent) ==========
if rank == 0:
    print("\n" + "="*50)
    print("ТЕСТ 1: Отложенные запросы (Persistent Operations)")
    print("="*50)

comm.Barrier()

# Подготовка данных
data_persistent = np.full(data_size, rank, dtype=np.int32)
data_recv_persistent = np.empty(data_size, dtype=np.int32)

# Создание отложенных запросов
persistent_requests = [MPI.REQUEST_NULL, MPI.REQUEST_NULL]
persistent_requests[0] = comm.Send_init([data_persistent, MPI.INT], dest=next_proc, tag=1)
persistent_requests[1] = comm.Recv_init([data_recv_persistent, MPI.INT], source=prev_proc, tag=1)

# Измерение времени
comm.Barrier()
start_persistent = MPI.Wtime()

for i in range(iterations):
    MPI.Prequest.Startall(persistent_requests)
    MPI.Request.Waitall(persistent_requests)
    # Обновление данных
    np.copyto(data_persistent, data_recv_persistent)

comm.Barrier()
end_persistent = MPI.Wtime()

# Освобождение ресурсов
for req in persistent_requests:
    req.Free()

time_persistent = end_persistent - start_persistent

# ========== ТЕСТ 2: Sendrecv_replace ==========
if rank == 0:
    print("\n" + "="*50)
    print("ТЕСТ 2: Sendrecv_replace (комбинированная операция)")
    print("="*50)

comm.Barrier()

# Подготовка данных
data_sendrecv = np.full(data_size, rank, dtype=np.int32)

# Измерение времени
comm.Barrier()
start_sendrecv = MPI.Wtime()

for i in range(iterations):
    # Единая операция: отправляет и принимает, заменяя данные
    comm.Sendrecv_replace(
        [data_sendrecv, MPI.INT], 
        dest=next_proc, 
        sendtag=2,
        source=prev_proc, 
        recvtag=2
    )

comm.Barrier()
end_sendrecv = MPI.Wtime()

time_sendrecv = end_sendrecv - start_sendrecv

# ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
if rank == 0:
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print("="*60)
    print(f"Количество процессов: {numprocs}")
    print(f"Размер данных: {data_size} элементов (int32)")
    print(f"Количество итераций: {iterations}")
    print("-" * 60)
    print(f"Время PERSISTENT операций:  {time_persistent:.6f} сек")
    print(f"Время SENDRECV_REPLACE:     {time_sendrecv:.6f} сек")
    print("-" * 60)
    
    # Расчет ускорения
    if time_sendrecv > 0:
        speedup = time_persistent / time_sendrecv
        print(f"Ускорение SENDRECV_REPLACE: {speedup:.2f}x")
    
    # Вывод рекомендации
    print("\nВЫВОД:")
    print("1. Persistent операции выгодны при МНОГОКРАТНОМ повторении")
    print("   одинаковых операций (экономия на создании запросов)")
    print("2. Sendrecv_replace удобнее для РАЗОВЫХ операций")
    print("   (проще в использовании, меньше кода)")
    
    # Проверка корректности данных
    print("\nПРОВЕРКА КОРРЕКТНОСТИ:")
    print("После всех итераций каждый процесс должен иметь")
    print("значение из начального соседа")

# Синхронизация и вывод финальных значений
comm.Barrier()
print(f"\nПроцесс {rank}: финальное значение[0] = {data_persistent[0]}")