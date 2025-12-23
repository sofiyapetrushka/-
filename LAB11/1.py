from mpi4py import MPI 
import numpy as np

comm = MPI.COMM_WORLD 
numprocs = comm.Get_size() 
rank = comm.Get_rank() 

print(f"Процесс {rank}: старт (всего процессов: {numprocs})")

if numprocs == 1:
    print("Запущен только 1 процесс. Невозможно создать кольцо.")
    print("Запустите программу так: mpiexec -n 4 python script.py")
elif numprocs == 2:
    # Особый случай для 2 процессов
    a = np.array([rank], dtype=np.int32)
    b = np.array([-1], dtype=np.int32)
    
    requests = [MPI.REQUEST_NULL, MPI.REQUEST_NULL]
    
    # Процесс 0 отправляет 1, получает от 1
    # Процесс 1 отправляет 0, получает от 0
    if rank == 0:
        requests[0] = comm.Isend([a, MPI.INT], dest=1, tag=0)
        requests[1] = comm.Irecv([b, MPI.INT], source=1, tag=0)
    else:  # rank == 1
        requests[0] = comm.Isend([a, MPI.INT], dest=0, tag=0)
        requests[1] = comm.Irecv([b, MPI.INT], source=0, tag=0)
else:
    # Общий случай для 3+ процессов
    a = np.array([rank], dtype=np.int32)
    b = np.array([-1], dtype=np.int32)
    
    requests = [MPI.REQUEST_NULL, MPI.REQUEST_NULL]
    
    send_to = (rank - 1) % numprocs
    recv_from = (rank + 1) % numprocs
    
    requests[0] = comm.Isend([a, MPI.INT], dest=send_to, tag=0)
    requests[1] = comm.Irecv([b, MPI.INT], source=recv_from, tag=0)

# Вычисления (если процессов > 1)
if numprocs > 1:
    print(f"Процесс {rank}: делаю полезную работу...")
    # Имитация вычислений
    data = np.random.rand(10000)
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    MPI.Request.Waitall(requests)
    
    print(f"Процесс {rank}: отправил {a[0]}, получил {b[0]}")
    print(f"Процесс {rank}: вычисления: mean={mean_val:.4f}, std={std_val:.4f}")