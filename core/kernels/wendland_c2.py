import numpy as np


def wendland_c2_kernel(r, h, *, dim: int = 3):
    """
    Ядро Вендланда C2: W(|r|,h) = σ_d / h^d · (1-q)^4(1+4q)   (q≤1).

    Parameters
    ----------
    r   : (..., dim) array_like — вектор расстояния r = x_a − x_b
    h   : float                 — сглаживающий радиус
    dim : {1,2,3}               — размерность пространства

    Returns
    -------
    W   : ndarray — значение ядра
    """
    # σd : нормировка для d = 1,2,3  (∫WdV = 1)
    sigma = {1: 3 / 4,
             2: 7 / (4 * np.pi),
             3: 21 / (16 * np.pi)}[dim] / np.pow(h, dim)
    r = np.asarray(r, dtype=float)
    q = np.linalg.norm(r, axis=-1) / h  # q = r/h

    if q > 1:
        return 0.0
    phi = ((1 - q) ** 4) * (1 + 4 * q)
    return sigma * phi


def wendland_c2_grad(r, h, *, dim: int = 3):
    """
    Градиент ядра Вендланда C2: ∇_r W.

    Returns
    -------
    gradW : ndarray — (…)×dim массив (та же форма, что r)
    """
    r = np.asarray(r, dtype=float)
    q = np.linalg.norm(r, axis=-1, keepdims=True) / h
    if q == 0 or q > 1:
        return np.zeros(dim)
    sigma = {1: 3 / 4,
             2: 7 / (4 * np.pi),
             3: 21 / (16 * np.pi)}[dim] / np.pow(h, dim)

    # φ'(q) = d/dq[(1-q)^4(1+4q)] = -20 q (1-q)^3  (при q≤1)
    phi_prime = np.where(q <= 1.0, -20.0 * q * (1 - q)**3, 0.0)

    dWdq = sigma * phi_prime  # dW/dq
    gradW = np.where(q > 0,
                     dWdq / (h * q) * r,              # ∇W = dW/dq·∇q
                     0.0)

    return gradW
