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

# Задаём узлы интерполяции (Вариант 2)
x_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
y_nodes = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

# Строим интерполяционный полином на интервале [x0 - 0.5, xn + 0.5] с шагом 0.05
x_start = x_nodes[0] - 0.5
x_end = x_nodes[-1] + 0.5
x_plot = np.arange(x_start, x_end + 1e-9, 0.05)
y_lagrange = [lagrange(x, x_nodes, y_nodes) for x in x_plot]

print("\n--- Задание 1: Интерполяция Лагранжа ---")
print("Узлы: x =", x_nodes)
print("Значения: y =", y_nodes)
print(f"\nЗначения полинома на интервале [{x_start}, {x_end}] с шагом 0.05:")
print(f"{'x':<10} {'P(x)':<15}")
print("-" * 25)
for x_val, y_val in zip(x_plot, y_lagrange):
    print(f"{x_val:<10.2f} {y_val:<15.6f}")

# ============================================================
# ЗАДАНИЕ 2: Аппроксимация методом наименьших квадратов
# ============================================================

def least_squares(x_nodes, y_nodes, degree):
    """
    Строит аппроксимирующий полином заданной степени методом наименьших квадратов.
    Возвращает коэффициенты полинома (от младшей степени к старшей).
    """
    n = len(x_nodes)
    m = degree + 1
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(m):
        for j in range(m):
            A[i, j] = np.sum(x_nodes ** (i + j))
        b[i] = np.sum(y_nodes * (x_nodes ** i))
    coeff = np.linalg.solve(A, b)
    return coeff

degree = 2
coeff = least_squares(x_nodes, y_nodes, degree)

def approx_poly(x, coeff):
    return np.polyval(coeff[::-1], x)

y_approx = approx_poly(x_plot, coeff)

print("\n--- Задание 2: Аппроксимация МНК ---")
print(f"Степень полинома: {degree}")
print(f"Коэффициенты (от младшей к старшей): {coeff}")

# ============================================================
# ЗАДАНИЕ 3: Сравнение интерполяции и аппроксимации
# ============================================================

print("\n--- Задание 3: Сравнение ---")
y_interp_nodes = [lagrange(x, x_nodes, y_nodes) for x in x_nodes]
y_approx_nodes = approx_poly(x_nodes, coeff)
print(f"{'Узел x':<8} {'y (исх.)':<10} {'Интерполяция':<15} {'Аппроксимация':<15} {'Интерп. ош.':<15} {'Аппрокс. ош.':<15}")
print("-" * 80)
for x, y_true, y_int, y_app in zip(x_nodes, y_nodes, y_interp_nodes, y_approx_nodes):
    print(f"{x:<8.2f} {y_true:<10.2f} {y_int:<15.6f} {y_app:<15.6f} {abs(y_true-y_int):<15.2e} {abs(y_true-y_app):<15.6f}")

# ============================================================
# ЗАДАНИЕ 4: Визуализация
# ============================================================

print("\n--- Задание 4: Графики ---")
plt.figure(figsize=(8, 5))
plt.plot(x_nodes, y_nodes, 'ko', markersize=8, label='Узлы')
plt.plot(x_plot, y_lagrange, 'r-', linewidth=2, label='Интерполяция Лагранжа')
plt.plot(x_plot, y_approx, 'b--', linewidth=2, label=f'Аппроксимация (степень {degree})')
plt.title('Интерполяция и аппроксимация (Вариант 2)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('lab5_interp_approx.png', dpi=300, bbox_inches='tight')
print("График сохранён в файл: lab5_interp_approx.png")
plt.show()

# ============================================================
# ЗАДАНИЕ 5: Исследование феномена Рунге
# ============================================================

print("\n--- Задание 5: Феномен Рунге ---")
# Таблично заданная функция варианта 2: y = 2 - |x| на [-2, 2]
def f_var2(x):
    return 2.0 - np.abs(x)

n_values = [5, 8, 11, 15]
plt.figure(figsize=(8, 5))
x_true = np.linspace(-2, 2, 300)
y_true = f_var2(x_true)
plt.plot(x_true, y_true, 'k-', linewidth=2, label='Исходная функция (2 - |x|)')

for n in n_values:
    x_nodes_r = np.linspace(-2, 2, n)
    y_nodes_r = f_var2(x_nodes_r)
    y_interp_r = [lagrange(x, x_nodes_r, y_nodes_r) for x in x_true]
    plt.plot(x_true, y_interp_r, '--', label=f'Интерполяция n={n}')

plt.title('Феномен Рунге: интерполяция функции варианта 2')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1.0, 3.0)
plt.legend()
plt.grid(True)
plt.savefig('lab5_runge.png', dpi=300, bbox_inches='tight')
print("График сохранён в файл: lab5_runge.png")
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)