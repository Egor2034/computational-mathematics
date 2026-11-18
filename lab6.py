import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №6")
print("Вариант 2")
print("=" * 60)

# ============================================================
# ЗАДАНИЕ 1: Численное дифференцирование
# ============================================================

def f(x):
    return np.cos(x)

def df_analytical(x):
    return -np.sin(x)

def left_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def right_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 0.5
h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

print("\n--- ЗАДАНИЕ 1: Численное дифференцирование ---")
print(f"x0 = {x0}, точное значение f'(x0) = {df_analytical(x0):.8f}")
print(f"{'h':<10} {'Левая':<15} {'Правая':<15} {'Центральная':<15} {'Ошибка центр.':<15}")
print("-" * 75)

errors_central = []
for h in h_values:
    dl = left_diff(f, x0, h)
    dr = right_diff(f, x0, h)
    dc = central_diff(f, x0, h)
    err_c = abs(dc - df_analytical(x0))
    errors_central.append(err_c)
    print(f"{h:<10.0e} {dl:<15.8f} {dr:<15.8f} {dc:<15.8f} {err_c:<15.2e}")

# ============================================================
# ЗАДАНИЕ 2: Исследование погрешности
# ============================================================

print("\n--- ЗАДАНИЕ 2: Зависимость ошибки от h для центральной разности ---")
plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors_central, 'o-', color='red', linewidth=2)
plt.title('Погрешность центральной разности в зависимости от шага h')
plt.xlabel('Шаг h')
plt.ylabel('Абсолютная погрешность')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('lab6_diff_error.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# ЗАДАНИЕ 3: Численное интегрирование
# ============================================================

def rectangle_left(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))

def rectangle_right(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))

def rectangle_mid(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5 * f(a) + np.sum(f(x[1:-1])) + 0.5 * f(b))

def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 1:
            result += 4 * f(x[i])
        else:
            result += 2 * f(x[i])
    return h / 3 * result

a, b = 0.0, np.pi / 2
exact = np.sin(b) - np.sin(a)  # интеграл cos(x) от 0 до pi/2
n_values = [2, 4, 8, 16, 32, 64]

print("\n--- ЗАДАНИЕ 3: Численное интегрирование ---")
print(f"Точный интеграл cos(x) от 0 до pi/2 = {exact:.8f}")
print(f"{'n':<6} {'Лев. прям.':<12} {'Прав. прям.':<12} {'Сред. прям.':<12} {'Трапеции':<12} {'Симпсон':<12}")
print("-" * 75)

for n in n_values:
    il = rectangle_left(f, a, b, n)
    ir = rectangle_right(f, a, b, n)
    im = rectangle_mid(f, a, b, n)
    it = trapezoidal(f, a, b, n)
    isim = simpson(f, a, b, n)
    print(f"{n:<6} {il:<12.8f} {ir:<12.8f} {im:<12.8f} {it:<12.8f} {isim:<12.8f}")

# ============================================================
# ЗАДАНИЕ 4: Сравнение погрешностей
# ============================================================

print("\n--- ЗАДАНИЕ 4: Погрешности интегрирования ---")
print(f"{'n':<6} {'Ош. лев.':<15} {'Ош. прав.':<15} {'Ош. сред.':<15} {'Ош. трап.':<15} {'Ош. Симпсон':<15}")
print("-" * 85)

errors_left = []
errors_right = []
errors_mid = []
errors_trap = []
errors_simp = []
for n in n_values:
    il = rectangle_left(f, a, b, n)
    ir = rectangle_right(f, a, b, n)
    im = rectangle_mid(f, a, b, n)
    it = trapezoidal(f, a, b, n)
    isim = simpson(f, a, b, n)
    el = abs(il - exact)
    er = abs(ir - exact)
    em = abs(im - exact)
    et = abs(it - exact)
    es = abs(isim - exact)
    errors_left.append(el)
    errors_right.append(er)
    errors_mid.append(em)
    errors_trap.append(et)
    errors_simp.append(es)
    print(f"{n:<6} {el:<15.2e} {er:<15.2e} {em:<15.2e} {et:<15.2e} {es:<15.2e}")

plt.figure(figsize=(8, 5))
plt.loglog(n_values, errors_left, 'o-', label='Левые прямоугольники')
plt.loglog(n_values, errors_right, 's-', label='Правые прямоугольники')
plt.loglog(n_values, errors_mid, 'd-', label='Средние прямоугольники')
plt.loglog(n_values, errors_trap, '^-', label='Метод трапеций')
plt.loglog(n_values, errors_simp, '*-', label='Метод Симпсона')
plt.title('Сравнение погрешностей интегрирования')
plt.xlabel('Число разбиений n')
plt.ylabel('Абсолютная погрешность')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('lab6_int_errors.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)