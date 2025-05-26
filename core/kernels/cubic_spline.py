from __future__ import annotations

import numpy as np
from typing import Sequence


def cubic_spline_kernel(r: Sequence[float] | float | int, h: float, dim: int = 2):
    """
    Кубический B-сплайн-ядро W(|r|,h) с поддержкой 2 h.

    Parameters
    ----------
    r   : (..., dim) array_like — вектор расстояний r = x_a − x_b
    h   : float                — сглаживающий радиус
    dim : {1,2,3}              — размерность пространства

    Returns
    -------
    W   : ndarray — значение ядра
    """
    # σd — коэффициенты нормировки (∫WdV = 1)
    sigma = {1: 2 / 3,
             2: 10 / (7 * np.pi),
             3: 1 / np.pi}[dim] / np.pow(h, dim)
    if np.isscalar(r):
        q = r / h
    else:
        r = np.asarray(r, dtype=float)
        q = np.linalg.norm(r, axis=-1) / h  # q = r/h

    kernel = np.zeros_like(q)

    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    kernel[mask1] = sigma * (1 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3)
    kernel[mask2] = sigma * 0.25 * (2 - q[mask2])**3

    return kernel


def cubic_spline_grad(r: Sequence[float] | float | int, h: float, dim: int = 2):
    """
    Градиент кубического сплайна ∇_r W.

    Returns
    -------
    gradW : ndarray — (…)×dim массив (такая же форма, как r)
    """
    # σd — коэффициенты нормировки (∫WdV = 1)
    sigma = {1: 2 / 3,
             2: 10 / (7 * np.pi),
             3: 1 / np.pi}[dim] / np.pow(h, dim)
    r = np.asarray(r, dtype=float)
    q = np.linalg.norm(r, axis=-1, keepdims=True) / h
    if q == 0:
        return np.zeros(dim, dtype=float)

    dWdq = np.zeros_like(q)

    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    dWdq[mask1] = sigma * (-3*q[mask1] + 2.25*q[mask1]**2)  # dW/dq
    dWdq[mask2] = sigma * (-0.75*(2 - q[mask2])**2)

    # ∇W = (dW/dq)/(h) · r/|r|
    return dWdq / (h*q) * r


def discrete_M0(kernel, h, dx):
    q = np.arange(-2*np.ceil(h/dx), 2*np.ceil(h/dx)+1)  # узлы
    W = [kernel(np.abs(qi*dx), h, dim=1) for qi in q]
    return np.sum(W)*dx


if __name__ == "__main__":
    dx = 1 / 2048
    for kappa in (1.3, 1.4, 1.5, 1.6):
        h = kappa*dx     # dx = 1/N
        print(kappa, discrete_M0(cubic_spline_kernel, h, dx))

