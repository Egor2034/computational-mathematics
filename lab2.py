import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №2")
print("КАТЮШИН Е.С. 6312-100503D")
print("=" * 60)

# ЗАДАНИЕ 1: Вычисление норм матрицы
A = np.array([
    [4.0, 1.0, 0.0],
    [1.0, 4.0, 1.0],
    [0.0, 1.0, 4.0]
])

print("\nЗадание 1: Нормы матрицы")
print("Матрица A:")
print(A)

# Вычисляем нормы с помощью numpy
norm1 = np.linalg.norm(A, ord=1)          # 1-норма (максимум суммы по столбцам)
norm_inf = np.linalg.norm(A, ord=np.inf)  # Бесконечная норма (максимум суммы по строкам)
norm_fro = np.linalg.norm(A, ord='fro')   # Норма Фробениуса

print(f"1-норма:            {norm1:.4f}")
print(f"Бесконечная норма:  {norm_inf:.4f}")
print(f"Норма Фробениуса:   {norm_fro:.4f}")

# ЗАДАНИЕ 2: Обусловленность матрицы
# Вычисляем число обусловленности по трём нормам
cond_1 = np.linalg.cond(A, p=1)
cond_inf = np.linalg.cond(A, p=np.inf)
cond_fro = np.linalg.cond(A, p='fro')

print("\nЗадание 2: Число обусловленности")
print(f"cond_1 (по 1-норме):          {cond_1:.4f}")
print(f"cond_inf (по бесконечной):    {cond_inf:.4f}")
print(f"cond_fro (по норме Фробениуса): {cond_fro:.4f}")

# ЗАДАНИЕ 3: Решение СЛАУ методом Гаусса
# Вектор правой части b для Варианта 2: (1, 1, 1)^T
b = np.array([1.0, 1.0, 1.0])

def gauss_solve(A, b):
    # Создаём копии, чтобы не испортить исходные данные
    n = len(A)
    a = A.astype(float).copy()
    bb = b.astype(float).copy()

    # Прямой ход
    for i in range(n):
        # Поиск главного элемента в столбце i (начиная со строки i)
        max_row = i
        max_val = abs(a[i][i])
        for k in range(i + 1, n):
            if abs(a[k][i]) > max_val:
                max_val = abs(a[k][i])
                max_row = k
        
        # Перестановка строк (если главный элемент не на диагонали)
        if max_row != i:
            a[[i, max_row]] = a[[max_row, i]]
            bb[[i, max_row]] = bb[[max_row, i]]

        # Проверка на вырожденность
        if abs(a[i][i]) < 1e-15:
            raise ValueError("Матрица вырождена (главный элемент близок к нулю)")

        # Исключение элементов ниже главной диагонали
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            a[j][i:] -= factor * a[i][i:]
            bb[j] -= factor * bb[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (bb[i] - np.dot(a[i][i + 1:], x[i + 1:])) / a[i][i]
    
    return x

print("\nЗадание 3: Решение СЛАУ методом Гаусса")
print("Вектор правой части b:")
print(b)

# Решаем методом Гаусса
x_gauss = gauss_solve(A, b)
print("Решение методом Гаусса (с выбором главного элемента):")
print(x_gauss)

# Сравниваем со встроенным решателем numpy
x_numpy = np.linalg.solve(A, b)
print("Решение встроенным методом numpy.linalg.solve:")
print(x_numpy)

# Оцениваем погрешность (бесконечная норма разности решений)
error = np.linalg.norm(x_gauss - x_numpy, ord=np.inf)
print(f"Максимальная погрешность (бесконечная норма): {error:.2e}")

# ЗАДАНИЕ 4: Исследование плохо обусловленных матриц
print("\nЗадание 4: Исследование плохо обусловленных матриц")

# Функция создания матрицы Гильберта (плохо обусловлена)
def hilbert(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H

sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
cond_numbers = []

for n in sizes:
    H = hilbert(n)
    cond = np.linalg.cond(H, p=np.inf)
    cond_numbers.append(cond)
    print(f"n = {n:2d}: cond(H) = {cond:.2e}")

# Строим график роста числа обусловленности
plt.figure(figsize=(8, 5))
plt.semilogy(sizes, cond_numbers, 'o-', color='red', linewidth=2)
plt.title('Рост числа обусловленности матрицы Гильберта')
plt.xlabel('Размер матрицы n')
plt.ylabel('Число обусловленности (логарифм. шкала)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.show()