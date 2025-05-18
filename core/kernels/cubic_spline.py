import numpy as np


def cubic_spline_kernel(r, h, dim: int = 3):
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
    r = np.asarray(r, dtype=float)
    q = np.linalg.norm(r, axis=-1) / h  # q = r/h

    kernel = np.zeros_like(q)

    mask1 = q < 1.0
    mask2 = (q >= 1.0) & (q < 2.0)

    kernel[mask1] = sigma * (1 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3)
    kernel[mask2] = sigma * 0.25 * (2 - q[mask2])**3

    return kernel


def cubic_spline_grad(r, h, dim: int = 3):
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


if __name__ == "__main__":
    ra = np.array([0.0, 0.0, 0.0])
    rb = np.array([0.0, 0.0, 0.0])
    dr = ra - rb

    W_val = cubic_spline_kernel(dr, h=0.2, dim=3)
    grad_W = cubic_spline_grad(dr, h=0.2, dim=3)

    print(f"W = {W_val:.4e}")
    print("∇W =", grad_W)
