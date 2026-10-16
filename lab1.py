import numpy as np
import matplotlib.pyplot as plt
import math

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №1")
print("КАТЮШИН Е.С. 6312-100503D")
print("=" * 60)


#ЗАДАНИЕ 1
def cosc(x):    
    if abs(x) < 1e-300:
        return 0.5
    return (1 - np.cos(x)) / (x * x)

# Создаем массив из 15 чисел от 10^-1 до 10^-15 для тестирования
x_vals = np.logspace(-1, -15, 15)

print("\nЗадание 1: Сравнение float64 и float32")
print(f"{'x':>10} {'float64':>20} {'float32':>20} {'Отн. погр.':>15}")
print("-" * 70)

for x in x_vals:
    res64 = cosc(np.float64(x)) # Вычисляем значение в двойной точности (эталон)
    res32 = cosc(np.float32(x)) # Вычисляем значение в одинарной точности

    # Расчет относительной погрешности
    if res64 != 0:
        rel_err = abs(res64 - res32) / abs(res64)
    else:
        rel_err = 0
    print(f"{x:>10.1e} {res64:>20.10f} {res32:>20.10f} {rel_err:>15.2e}")


#ЗАДАНИЕ 2
def machine_epsilon(dtype):
    eps = 1.0
    while dtype(1.0 + eps) != dtype(1.0):
        eps /= 2.0
    return eps * 2.0


print("\nЗадание 2: Машинный эпсилон")
print(f"float64: {machine_epsilon(np.float64):.5e}")
print(f"float32: {machine_epsilon(np.float32):.5e}")


#ЗАДАНИЕ 3
def taylor_cos(x, eps):
    term = 1.0 # Первый член ряда
    summa = 1.0 # Сумма, начинается с первого члена
    n = 1 # Номер следующего члена

    while abs(term) > eps:
        # Рекуррентная формула: следующий член через предыдущий
        term *= (-x * x) / ((2 * n - 1) * 2 * n)
        summa += term
        n += 1

    return summa, n - 1


print("\nЗадание 3: Ряд Тейлора для cos(x)")
for x in [0.5, 1.0, 2.0]:
    for eps in [1e-6, 1e-10]:
        approx, terms = taylor_cos(x, eps)
        exact = math.cos(x)
        print(f"x={x:3.1f}, eps={eps:.0e}: approx={approx:.10f}, "
            f"exact={exact:.10f}, terms={terms:2d}, error={abs(approx - exact):.2e}")


#ЗАДАНИЕ 4
print("\n--- Задание 4: График погрешности ---")
x_const = 3.0
exact_val = math.cos(x_const)
terms_list = []
errors_list = []

term = 1.0
summa = 1.0
n = 1

while n <= 30:
    term *= (-x_const * x_const) / ((2 * n - 1) * 2 * n)
    summa += term
    terms_list.append(n)
    errors_list.append(abs(exact_val - summa))
    n += 1

plt.figure(figsize=(8, 5))
plt.semilogy(terms_list, errors_list, 'o-', color='red', linewidth=2)
plt.title('Погрешность частичных сумм для cos(3)')
plt.xlabel('Число членов ряда')
plt.ylabel('Абсолютная погрешность (лог. шкала)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.show()