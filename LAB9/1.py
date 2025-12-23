from numpy import empty, linspace, sin, pi, float64, save
import time

# Параметры задачи
eps = 0.01
L = 1.0          # длина области
T = 1.0          # конечное время
N = 20000        # число узлов по пространству (включая границы)
M = 5000         # число шагов по времени
h = L / (N - 1)  # шаг по пространству
tau = T / M      # шаг по времени
alpha = 1.0      # коэффициент ROS1

# Граничные условия
def u_left(t):
    return sin(pi * t)

def u_right(t):
    return 0.0

# Правая часть системы: f(y, t, ...)
def f(y, t, h, N, u_left, u_right, eps):
    f_arr = empty(N - 1, dtype=float64)
    # Первый внутренний узел (n=0)
    f_arr[0] = (
        eps * (y[1] - 2 * y[0] + u_left(t)) / h**2 +
        y[0] * (y[1] - u_left(t)) / (2 * h) +
        y[0]**3
    )
    # Внутренние узлы
    for n in range(1, N - 2):
        f_arr[n] = (
            eps * (y[n+1] - 2 * y[n] + y[n-1]) / h**2 +
            y[n] * (y[n+1] - y[n-1]) / (2 * h) +
            y[n]**3
        )
    # Последний внутренний узел (n = N-2)
    f_arr[N-2] = (
        eps * (u_right(t) - 2 * y[N-2] + y[N-3]) / h**2 +
        y[N-2] * (u_right(t) - y[N-3]) / (2 * h) +
        y[N-2]**3
    )
    return f_arr

# Формирование трёхдиагональной матрицы (I - alpha*tau*J)
def diagonal_preparation(y, t, h, N, u_left, u_right, eps, tau, alpha):
    a = empty(N - 1, dtype=float64)  # поддиагональ
    b = empty(N - 1, dtype=float64)  # диагональ
    c = empty(N - 1, dtype=float64)  # наддиагональ

    # Первый узел
    b[0] = 1.0 - alpha * tau * (
        -2 * eps / h**2 +
        (y[1] - u_left(t)) / (2 * h) +
        3 * y[0]**2
    )
    c[0] = -alpha * tau * (eps / h**2 + y[0] / (2 * h))

    # Внутренние узлы
    for n in range(1, N - 2):
        a[n] = -alpha * tau * (eps / h**2 - y[n] / (2 * h))
        b[n] = 1.0 - alpha * tau * (
            -2 * eps / h**2 +
            (y[n+1] - y[n-1]) / (2 * h) +
            3 * y[n]**2
        )
        c[n] = -alpha * tau * (eps / h**2 + y[n] / (2 * h))

    # Последний узел
    a[N-2] = -alpha * tau * (eps / h**2 - y[N-2] / (2 * h))
    b[N-2] = 1.0 - alpha * tau * (
        -2 * eps / h**2 +
        (u_right(t) - y[N-3]) / (2 * h) +
        3 * y[N-2]**2
    )
    # c[N-2] не используется (но задаём 0 для безопасности)
    c[N-2] = 0.0

    return a, b, c

# Метод прогонки (для СЛАУ A x = d, где A — трёхдиагональная)
def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N_sys = len(d)
    x = empty(N_sys, dtype=float64)

    # Прямой ход
    for n in range(1, N_sys):
        coef = a[n] / b[n-1]
        b[n] = b[n] - coef * c[n-1]
        d[n] = d[n] - coef * d[n-1]

    # Обратный ход
    x[N_sys - 1] = d[N_sys - 1] / b[N_sys - 1]
    for n in range(N_sys - 2, -1, -1):
        x[n] = (d[n] - c[n] * x[n + 1]) / b[n]

    return x

# Основная программа
def main():
    # Инициализация пространственной сетки
    x_grid = linspace(0, L, N)
    # Начальное условие (внутренние узлы: без границ)
    y = sin(pi * x_grid[1:-1]).astype(float64)  # длина N-2, но у нас N-1 вектор? → исправим

    # Пересчитаем параметры в соответствии с лекцией:

    global h, N
    N_nodes = N  # из условия — это N из лекции (внутренних узлов = N-1)
    h = L / N_nodes
    x_inner = linspace(h, L - h, N_nodes - 1)  # внутренние узлы
    y = sin(pi * x_inner).astype(float64)      # начальное условие на внутренних узлах

    # Временная интеграция
    start_time = time.time()
    for m in range(M):
        t = m * tau

        # 1. Вычисляем правую часть f(y, t)
        rhs = f(y, t, h, N_nodes, u_left, u_right, eps)

        # 2. Формируем матрицу (I - alpha*tau*J) → трёхдиагональные коэффициенты
        a, b, c = diagonal_preparation(y, t, h, N_nodes, u_left, u_right, eps, tau, alpha)

        # 3. Решаем СЛАУ: (I - alpha*tau*J) * k = rhs
        k = consecutive_tridiagonal_matrix_algorithm(a, b, c, rhs)

        # 4. Обновляем решение: y_new = y + tau * k
        y += tau * k

    elapsed = time.time() - start_time
    print(f"Время выполнения (N={N}, M={M}): {elapsed:.2f} секунд")

    # Сохраняем результат (внутренние узлы + границы)
    u_full = empty(N_nodes + 1, dtype=float64)
    u_full[0] = u_left(T)          # левая граница в момент T
    u_full[1:-1] = y               # внутренние узлы
    u_full[-1] = u_right(T)        # правая граница

    # Сохраняем в файл
    save("solution_ros1.npy", u_full)
    print("Решение сохранено в 'solution_ros1.npy'")

if __name__ == "__main__":
    main()