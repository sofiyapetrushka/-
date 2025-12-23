from numpy import empty, linspace, tanh, savez
import time

# Начальное условие
def u_init(x, y):
    return 0.5 * tanh(1 / eps * ((x - 0.5)**2 + (y - 0.5)**2 - 0.35**2)) - 0.17

# Граничные условия (все константы = 0.33)
def u_left(y, t):    return 0.33
def u_right(y, t):   return 0.33
def u_top(x, t):     return 0.33
def u_bottom(x, t):  return 0.33

# Параметры из лекции (обновлены по заданию)
a, b = -2.0, 2.0
c, d = -2.0, 2.0
t_0, T = 0.0, 4.0
eps = 10**(-1.5)  # ≈ 0.03162
N_x, N_y = 200, 200
M = 4000

# Сетка
x, h_x = linspace(a, b, N_x + 1, retstep=True)
y, h_y = linspace(c, d, N_y + 1, retstep=True)
t, tau = linspace(t_0, T, M + 1, retstep=True)

# Массив решения: u[время, i, j]
u = empty((M + 1, N_x + 1, N_y + 1))

# Инициализация начального условия
for i in range(N_x + 1):
    for j in range(N_y + 1):
        u[0, i, j] = u_init(x[i], y[j])

# Установка граничных условий на начальный момент
for j in range(N_y + 1):
    u[0, 0, j] = u_left(y[j], t_0)
    u[0, N_x, j] = u_right(y[j], t_0)
for i in range(N_x + 1):
    u[0, i, 0] = u_bottom(x[i], t_0)
    u[0, i, N_y] = u_top(x[i], t_0)

# Основной цикл по времени
start_time = time.time()
for m in range(M):
    t_current = t[m]

    # Граничные условия на новом слое (явная схема — можно задать сразу)
    for j in range(N_y + 1):
        u[m + 1, 0, j] = u_left(y[j], t_current + tau)
        u[m + 1, N_x, j] = u_right(y[j], t_current + tau)
    for i in range(N_x + 1):
        u[m + 1, i, 0] = u_bottom(x[i], t_current + tau)
        u[m + 1, i, N_y] = u_top(x[i], t_current + tau)

    # Внутренние узлы
    for i in range(1, N_x):
        for j in range(1, N_y):
            d2x = (u[m, i + 1, j] - 2 * u[m, i, j] + u[m, i - 1, j]) / h_x**2
            d2y = (u[m, i, j + 1] - 2 * u[m, i, j] + u[m, i, j - 1]) / h_y**2
            d1x = (u[m, i + 1, j] - u[m, i - 1, j]) / (2 * h_x)
            d1y = (u[m, i, j + 1] - u[m, i, j - 1]) / (2 * h_y)
            u[m + 1, i, j] = u[m, i, j] + tau * (
                eps * (d2x + d2y) +
                u[m, i, j] * (d1x + d1y) +
                u[m, i, j]**3
            )

elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed:.4f} sec")

# Сохранение для последующей визуализации/анимации
savez("results_2d.npz", x=x, y=y, t=t, u=u)
print("Результаты сохранены в 'results_2d.npz'")