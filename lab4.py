import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №4")
print("КАТЮШИН Е.С. 6312-100503D")
print("Вариант 2")
print("=" * 60)

# ЗАДАНИЕ 1: Решение системы методом Ньютона
# Система для Варианта 2:
# x^2 + y^2 = 1
# x^2 - y^2 = 0
# Переносим все в левую часть:
# f1 = x^2 + y^2 - 1
# f2 = x^2 - y^2
def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1,
        x[0]**2 - x[1]**2
    ])

def J(x):
    """
    Матрица Якоби для системы.
    df1/dx = 2x,  df1/dy = 2y
    df2/dx = 2x,  df2/dy = -2y
    """
    return np.array([
        [2*x[0], 2*x[1]],
        [2*x[0], -2*x[1]]
    ])

def newton_system(F, J, x0, eps=1e-8, max_iter=100):
    """
    Решает систему F(x)=0 методом Ньютона.
    x0 - начальное приближение (вектор).
    Возвращает (корень, число итераций, список приближений).
    """
    x = x0.astype(float).copy()
    history = [x.copy()]
    for i in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < eps:
            return x, i, history
        Jx = J(x)
        if abs(np.linalg.det(Jx)) < 1e-15:
            raise ValueError("Матрица Якоби вырождена")
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i + 1, history
        x = x_new
    raise RuntimeError("Метод Ньютона не сошелся за заданное число итераций")

# Начальное приближение для Варианта 2: (0.8, 0.5)
x0 = np.array([0.8, 0.5])
root, iterations, history = newton_system(F, J, x0)
print("\n--- Задание 1: Решение системы методом Ньютона ---")
print(f"Начальное приближение: x0 = {x0}")
print(f"Найденный корень: x = {root}")
print(f"Число итераций: {iterations}")
print(f"Невязка ||F(x)||: {np.linalg.norm(F(root)):.2e}")

# ЗАДАНИЕ 2: Проверка точности
print("\n--- Задание 2: Проверка с помощью scipy.optimize.fsolve ---")
try:
    from scipy.optimize import fsolve
    root_scipy = fsolve(F, x0)
    print(f"Решение scipy: {root_scipy}")
    print(f"Невязка: {np.linalg.norm(F(root_scipy)):.2e}")
    print(f"Разница: {np.linalg.norm(root - root_scipy):.2e}")
except ImportError:
    print("scipy не установлен. Пропускаем.")

# ЗАДАНИЕ 3: Исследование сходимости
print("\n--- Задание 3: Исследование сходимости ---")
print(f"{'Нач. x':<10} {'Нач. y':<10} {'Итераций':<10} {'Корень x':<12} {'Корень y':<12}")
print("-" * 70)

starts = [
    (1.5, 1.5),
    (1.0, 1.0),
    (2.0, 1.0),
    (-1.5, -1.0),
    (0.5, -0.5)
]

for x0, y0 in starts:
    try:
        x0_vec = np.array([x0, y0])
        root_tmp, it_tmp, _ = newton_system(F, J, x0_vec)
        res_tmp = np.linalg.norm(F(root_tmp))
        print(f"{x0:<10.2f} {y0:<10.2f} {it_tmp:<10} {root_tmp[0]:<12.6f} {root_tmp[1]:<12.6f}")
    except Exception as e:
        print(f"{x0:<10.2f} {y0:<10.2f} {'не сошелся':<10} {'-':<12} {'-':<12}")

# ЗАДАНИЕ 4: Визуализация
print("\n--- Задание 4: График сходимости ---")
# Строим линии уровня для системы
x_vals = np.linspace(-2.5, 2.5, 100)
y_vals = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = X**2 + Y**2 - 1
F2 = X**2 - Y**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2, label='x^2+y^2=1')
plt.contour(X, Y, F2, levels=[0], colors='green', linewidths=2, label='x^2-y^2=0')

# Отображаем итерации
history = np.array(history)
plt.plot(history[:, 0], history[:, 1], 'ro-', linewidth=1.5, markersize=4, label='Итерации')
plt.plot(root[0], root[1], 'ks', markersize=8, label='Корень')

plt.title('Метод Ньютона для системы нелинейных уравнений')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)