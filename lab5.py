import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №5")
print("КАТЮШИН Е.С. 6312-100503D")
print("Вариант 2")
print("=" * 60)

# ============================================================
# ЗАДАНИЕ 1: Интерполяционный полином Лагранжа
# ============================================================

def lagrange(x, x_nodes, y_nodes):
    """
    Вычисляет значение интерполяционного полинома Лагранжа в точке x.
    x_nodes, y_nodes - массивы узлов и значений.
    """
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return result

# Данные для Варианта 2
x_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y_nodes = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

# Интервал [x0 - 0.5, xn + 0.5] с шагом 0.05
x_step = np.arange(x_nodes[0] - 0.5, x_nodes[-1] + 0.5 + 1e-9, 0.05)
y_lagrange_step = [lagrange(x, x_nodes, y_nodes) for x in x_step]

print("\n--- Задание 1: Интерполяция Лагранжа ---")
print("Узлы: x =", x_nodes)
print("Значения: y =", y_nodes)
print(f"\nЗначения полинома на интервале [{x_nodes[0]-0.5}, {x_nodes[-1]+0.5}] с шагом 0.05:")
print(f"{'x':<10} {'P(x)':<15}")
print("-" * 25)
for x_val, y_val in zip(x_step, y_lagrange_step):
    print(f"{x_val:<10.2f} {y_val:<15.6f}")

# ============================================================
# ЗАДАНИЕ 2: Аппроксимация методом наименьших квадратов
# ============================================================

def least_squares(x_nodes, y_nodes, degree):
    """
    Строит аппроксимирующий полином заданной степени методом наименьших квадратов.
    Возвращает коэффициенты полинома (от младшей степени к старшей).
    """
    m = degree + 1
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(m):
        for j in range(m):
            A[i, j] = np.sum(x_nodes ** (i + j))
        b[i] = np.sum(y_nodes * (x_nodes ** i))
    coeff = np.linalg.solve(A, b)
    return coeff

def approx_poly(x, coeff):
    return np.polyval(coeff[::-1], x)

degree = 2  # Степень аппроксимации для Варианта 2
coeff = least_squares(x_nodes, y_nodes, degree)
y_approx_step = approx_poly(x_step, coeff)

print("\n--- Задание 2: Аппроксимация МНК ---")
print(f"Степень полинома: {degree}")
print(f"Коэффициенты (от младшей к старшей): {coeff}")

# ============================================================
# ЗАДАНИЕ 3: Сравнение интерполяции и аппроксимации
# ============================================================

print("\n--- Задание 3: Сравнение в узлах ---")
y_interp_nodes = [lagrange(x, x_nodes, y_nodes) for x in x_nodes]
y_approx_nodes = approx_poly(x_nodes, coeff)

print(f"{'Узел x':<8} {'y (исх.)':<10} {'Интерполяция':<15} {'Аппроксимация':<15} {'Ош. интерп.':<14} {'Ош. аппрокс.':<14}")
print("-" * 80)
for x, y_true, y_int, y_app in zip(x_nodes, y_nodes, y_interp_nodes, y_approx_nodes):
    err_int = abs(y_true - y_int)
    err_app = abs(y_true - y_app)
    print(f"{x:<8.2f} {y_true:<10.2f} {y_int:<15.6f} {y_app:<15.6f} {err_int:<14.2e} {err_app:<14.6f}")

# ============================================================
# ЗАДАНИЕ 4: Визуализация
# ============================================================

print("\n--- Задание 4: График интерполяции и аппроксимации ---")
plt.figure(figsize=(8, 5))
plt.plot(x_nodes, y_nodes, 'ko', markersize=8, label='Узлы')
plt.plot(x_step, y_lagrange_step, 'r-', linewidth=2, label='Интерполяция Лагранжа')
plt.plot(x_step, y_approx_step, 'b--', linewidth=2, label=f'Аппроксимация МНК (степень {degree})')
plt.title('Интерполяция и аппроксимация (Вариант 2)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# ЗАДАНИЕ 5: Исследование феномена Рунге
# ============================================================

print("\n--- Задание 5: Исследование феномена Рунге ---")
# Для таблично заданной функции f(x) = 2 - |x| (соответствует узлам варианта 2)
def f_variant2(x):
    return 2.0 - np.abs(x)

n_values = [3, 5, 9, 13]
plt.figure(figsize=(8, 5))
x_true = np.linspace(-2, 2, 300)
y_true = f_variant2(x_true)
plt.plot(x_true, y_true, 'k-', linewidth=2, label='Исходная функция (2 - |x|)')

for n in n_values:
    x_nodes_r = np.linspace(-2, 2, n)
    y_nodes_r = f_variant2(x_nodes_r)
    y_interp_r = [lagrange(x, x_nodes_r, y_nodes_r) for x in x_true]
    plt.plot(x_true, y_interp_r, '--', label=f'Интерполяция n={n}')

plt.title('Исследование феномена Рунге на равномерной сетке (Вариант 2)')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.0, 3.0)
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)