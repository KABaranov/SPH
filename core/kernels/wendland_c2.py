from __future__ import annotations

import numpy as np
from typing import Sequence


def wendland_c2_kernel(r: Sequence[float] | float | int, h: float, dim: int = 2):
    """
    Ядро Вендланда C2: W(|r|,h) = σ_d / h^d · (1-q)^4(1+4q)   (q≤1).

    Parameters
    ----------
    r   : (..., dim) array_like — вектор расстояния r = x_a − x_b
    h   : float                 — сглаживающий радиус
    dim : {1,2,3}               — размерность пространства

    Returns
    -------
    W   : float — значение ядра
    """
    if np.isscalar(r):
        q = r / h
    else:
        r = np.asarray(r, dtype=float)
        q = np.linalg.norm(r, axis=-1) / h  # q = r/h
    if q > 2.0:
        return 0.0
    sigma = {1: 3 / 4,
             2: 7 / (4 * np.pi),
             3: 21 / (16 * np.pi)}[dim] / np.pow(h, dim)
    phi = (1 - 0.5*q)**4 * (2*q + 1)
    return sigma * phi


def wendland_c2_grad(r: Sequence[float] | float | int, h: float, dim: int = 3):
    """
    Градиент ядра Вендланда C2: ∇_r W.

    Returns
    -------
    gradW : ndarray — (…)×dim массив (та же форма, что r)
    """
    sigma = {1: 3 / 4,
             2: 7 / (4 * np.pi),
             3: 21 / (16 * np.pi)}[dim] / np.pow(h, dim)
    r = np.asarray(r, dtype=np.float64)

    # --- если подан модуль, добавим фиктивную компоненту для унификации -------
    scalar_input = (r.ndim == 0) or (r.ndim == 1 and r.shape[-1] != dim)
    if scalar_input:
        # |r| → вектор нулей той же размерности (градиент = 0)
        r_vec = np.zeros(r.shape + (dim,))
        r_mag = np.asarray(r)
    else:
        r_vec = r
        r_mag = np.linalg.norm(r_vec, axis=-1)

    q = r_mag / h
    a = 1.0 - 0.5 * q           # вспомогательное (1 - q/2)

    # радиальная производная: dW/dr = σ_d / h^{d+1} * (-5 q a^3)   (для q<2)
    dw_dq = -5.0 * q * a**3
    factor = sigma / h**(dim + 1)
    dW_dr = np.where(q < 2.0, factor * dw_dq, 0.0)

    # векторный градиент: ∇W = dW/dr * r̂
    # безопасно обрабатываем r=0
    with np.errstate(divide='ignore', invalid='ignore'):
        gradW = (dW_dr[..., None] * r_vec) / r_mag[..., None]
        gradW = np.nan_to_num(gradW)   # заменяем NaN при r=0 на 0

    # если вход был скалярным |r| — вернём нули такой же формы
    if scalar_input:
        gradW = np.zeros(r_mag.shape + (dim,))

    return gradW
