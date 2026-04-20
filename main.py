import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x)


# по факту в аналитике получался минимум
# так что здесь мы тоже берём просто минимум на [x_k-1, x_k] без проверки на принадлежность x > или < pi
def value_at_xk(a, k, h):
    left = a + k * h
    right = left + h

    return np.minimum(np.cos(left), np.cos(right))


def f_n(x, n):
    a, b = 0.0, 4.0
    h = (b - a) / n
    x = np.array(x, dtype=float)

    k = np.floor((x - a) / h).astype(int)
    k = np.clip(k, 0, n - 1)

    return value_at_xk(a, k, h)


def plot_fn_examples():
    x = np.linspace(0, 4, 2000)

    plt.figure(figsize=(10, 6))

    plt.plot(x, np.cos(x), "k", linewidth=2, label="cos(x)")

    for n in [25, 75, 150]:
        plt.step(x, f_n(x, n), where="mid", label=f"f_n, n={n}")

    plt.title("Приближение cos(x) функциями f_n")
    plt.legend()
    plt.grid()
    plt.show()


def lebesgue_integral_fn(n):
    a, b = 0.0, 4.0
    h = (b - a) / n

    res = 0.0

    for k in range(n):
        res += value_at_xk(a, k, h) * h

    return res


def lebesgue_stieltjes_integral_fn(n):
    points = np.arange(1, 9) / 2
    # да , умножение на 1 странно , просто хотел показать , что меру мы тут учитываем (просто по аналитике она 1)
    return np.sum(f_n(points, n) * 1)


def calc_and_print_integral(analitics_value, integral_func, n_to_iterate, name):
    print(f"Аналитиченское решение для {name}:", analitics_value)
    print(f"Числовое решение для {name} с разными значениями n:")
    for n in n_to_iterate:
        value = integral_func(n)
        offset = len(str(n_to_iterate[-1])) - len(str(n))
        print(
            f"n={n} {' ' * offset}: {value:.6f}, ошибка = {abs(value - analitics_value):.6f}"
        )
    print()


# для 2.1
# пытался подобрать занчения для n получше , но получилось так (всё равно можно увеличить)
plot_fn_examples()

# для 2.2 и 2.3
# взял 10^4 и 10^5 , т.к. там уже ошибка на уровне тысячных и десятетысячных , думаю , что норм
calc_and_print_integral(
    np.sin(4), lebesgue_integral_fn, [10, 100, 1000, 10_000, 100_000], "пункта 2.2"
)
calc_and_print_integral(
    np.sum(np.cos(np.arange(1, 9) / 2)),
    lebesgue_stieltjes_integral_fn,
    [10, 100, 1000, 10_000, 100_000],
    "пункта 2.3",
)
