from __future__ import annotations

import numpy as np
from typing import Sequence


def gaussian_kernel(r: Sequence[float] | float | int, h: float,
                    dim: int = 2, rcut: float = -1.0) -> float:
    """
    Усечённое гауссово ядро W(|r|, h).

    Parameters
    ----------
    r    : (..., dim) array_like — вектор расстояния r = x_a − x_b
    h    : float               — сглаживающий радиус
    dim  : {1,2,3}             — размерность пространства
    rcut : float               — радиус поддержки в единицах h (по умолчанию 3 h)

    Returns
    -------
    kernel    : ndarray — значение ядра в точках r
    """
    # нормирующие коэффициенты σd  (∫WdV = 1)
    sigma = 1.0 / np.pow(np.pi, dim / 2.0)
    if np.isscalar(r):
        q = r / h
    else:
        r = np.asarray(r, dtype=float)
        q = np.linalg.norm(r, axis=-1) / h  # q = r/h
    if (rcut != -1) and (q > rcut):
        return 0
    coeff = sigma / np.pow(h, dim)
    kernel = coeff / np.exp(q ** 2.0)
    return kernel


def gaussian_grad(r: Sequence[float] | float | int, h: float,
                  dim: int = 2, rcut: float = 3.0):
    """
    Градиент гауссова ядра ∇_r W (вектор той же размерности, что r).

    Returns
    -------
    gradW : ndarray — (…)×dim массив, совпадающий по форме с r
    """
    r = np.asarray(r, dtype=float)
    kernel = gaussian_kernel(r, h, dim=dim, rcut=rcut)
    if np.linalg.norm(r, axis=-1, keepdims=True) > rcut * h:
        return r * 0.0
    return -2.0 * kernel * r / np.pow(h, 2.0)  # ∇W = -2 W r / h²


if __name__ == "__main__":
    from cubic_spline import discrete_M0
    dx = 1 / 2048
    for kappa in (1.3, 1.4, 1.5, 1.6):
        h = kappa*dx     # dx = 1/N
        print(kappa, discrete_M0(gaussian_kernel, h, dx))
