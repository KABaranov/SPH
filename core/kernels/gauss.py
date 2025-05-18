import numpy as np


def gaussian_kernel(r, h, dim: int = 3, rcut: float = 3.0):
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
    r = np.asarray(r, dtype=float)
    q = np.linalg.norm(r, axis=-1) / h  # q = r/h
    if q > rcut:
        return 0
    coeff = sigma / np.pow(h, dim)
    kernel = coeff / np.exp(q ** 2.0)
    return kernel


def gaussian_grad(r, h, dim: int = 3, rcut: float = 3.0):
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
    ra = np.array([0.0, 0.0, 0.0])
    rb = np.array([0.0, 0.0, 0.0])
    dr = ra - rb

    W_val = gaussian_kernel(dr, h=0.2, dim=3)
    grad_W = gaussian_grad(dr, h=0.2, dim=3)

    print(f"W = {W_val:.4e}")
    print("∇W =", grad_W)

