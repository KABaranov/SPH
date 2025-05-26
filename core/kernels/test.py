import numpy as np
import matplotlib.pyplot as plt
from gauss import gaussian_kernel, gaussian_grad
from cubic_spline import cubic_spline_kernel, cubic_spline_grad
from wendland_c2 import wendland_c2_kernel, wendland_c2_grad

from typing import Callable, List


def plot_kernel(kernel_func: Callable, h: float = 1.0, dim: int = 3,
                rcut: float = 3.0, n: int = 400):
    q = np.linspace([-rcut], rcut, n)
    kernel_list = [kernel_func(qi, h, dim=dim) for qi in q]
    plt.figure()
    plt.plot(q, kernel_list)
    plt.xlabel(r'$q = r/h$')
    plt.ylabel(r'$W(q)$')
    plt.title('Kernel')
    plt.grid(True)


def plot_grad(grad_func: Callable, h: float = 1.0, dim: int = 3,
              rcut: float = 3.0, n: int = 400):
    q = np.linspace([-rcut], rcut, n)
    grad_list = [grad_func(qi, h, dim=dim) for qi in q]
    plt.figure()
    plt.plot(q, grad_list)
    plt.xlabel(r'$q = r/h$')
    plt.ylabel(r'$\partial W / \partial r$')
    plt.title('Gradient')
    plt.grid(True)


def plot_many_kernel(kernel_dict: dict, h: float = 1.0,
                     dim: int = 3, rcut: float = 3.0, n: int = 400):
    q = np.linspace([-rcut], rcut, n)
    plt.figure()
    kernel_names = kernel_dict.keys()
    for kernel_name in kernel_names:
        kernel_func = kernel_dict[kernel_name]
        kernel_i = [kernel_func(qi, h, dim=dim) for qi in q]
        plt.plot(q, kernel_i)
    plt.legend(kernel_names)
    plt.xlabel(r'$q = r/h$')
    plt.ylabel(r'$W(q)$')
    plt.title(f'Сравнение ядер в {dim}D')
    plt.grid(True)


def plot_many_gradients(grad_dict: dict, h: float = 1.0,
                        dim: int = 3, rcut: float = 3.0, n: int = 400):
    q = np.linspace([-rcut], rcut, n)
    plt.figure()
    grad_names = grad_dict.keys()
    for grad_name in grad_names:
        grad_func = grad_dict[grad_name]
        grad_i = [grad_func(qi, h, dim=dim) for qi in q]
        plt.plot(q, grad_i)
    plt.legend(grad_names)
    plt.xlabel(r'$q = r/h$')
    plt.ylabel(r'$W(q)$')
    plt.title('Сравнение градиентов')
    plt.grid(True)


def kernel_m0(kernel: Callable, h: float, *, dim: int = 1,
              k_cut: float = 2.0,  # радиус усечения -- обычно 2 h
              n: int = 20000):  # число точек интегрирования
    """
    Вычисляет нулевой момент (нормировку) SPH-ядра.

    Parameters
    ----------
    kernel : callable
        Функция W(r, h), векторизуемая по r (NumPy).
    h : float
        Сглаживающий радиус ядра.
    dim : {1, 2, 3}
        Пространственная размерность.
    k_cut : float
        Компактная поддержка ядра в единицах h (для куб. сплайна = 2).
    n : int
        Число точек интегрирования по r.
    tol : float | None
        Если задано, проверяет |M0 - 1| < tol и бросает AssertionError
        при нарушении.

    Returns
    -------
    M0 : float
        Численно полученный нулевой момент.
    """
    r = np.linspace(0.0, k_cut * h, n, dtype=float)
    w = [kernel([ri], h, dim=dim) for ri in r]
    if dim == 1:
        integrand = w
        M0 = 2.0 * np.trapezoid(integrand, r)
    elif dim == 2:
        integrand = w * r
        M0 = 2.0 * np.pi * np.trapezoid(integrand, r)
    elif dim == 3:
        integrand = w * r ** 2
        M0 = 4.0 * np.pi * np.trapezoid(integrand, r)
    else:
        raise ValueError("dim must be 1, 2 or 3")

    return M0


if __name__ == "__main__":
    # plot_kernel(cubic_spline_kernel, dim=1)
    # plot_grad(cubic_spline_grad, dim=1)
    dx = 0.001
    kappa = 1.3
    h = kappa * dx
    print(kernel_m0(wendland_c2_kernel, h=h))
    print(kernel_m0(gaussian_kernel, h=h))
    print(kernel_m0(cubic_spline_kernel, h=h))
    kernels = {"Ядро Гаусса": gaussian_kernel,
               "Кубический Сплайн": cubic_spline_kernel,
               "Вендланд C2": wendland_c2_kernel}
    gradients = {"Ядро Гаусса": gaussian_grad,
                 "Кубический Сплайн": cubic_spline_grad,
                 "Вендланд C2": wendland_c2_grad}
    plot_many_kernel(kernels, dim=1)
    plot_many_gradients(gradients, dim=1)
    plt.show()
