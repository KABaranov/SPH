import numpy as np


def cubic_spline_kernel(r: np.ndarray, h: float, dim: int) -> float:
    """
    Cubic spline kernel for 1D, 2D, or 3D, always accepts r as length-3 vector.

    Parameters
    ----------
    r   : ndarray, shape (3,)  - displacement vector between two particles
    h   : float               - smoothing length
    dim : int                 - dimensionality (1, 2, or 3)

    Returns
    -------
    W : float                 - kernel value
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
    r3 = np.asarray(r, dtype=float)
    if r3.shape != (3,):
        raise ValueError(f"Input r must have shape (3,), got {r3.shape}")
    # consider only first dim components
    r_vec = r3[:dim]
    r_norm = np.linalg.norm(r_vec)
    q = r_norm / h
    # normalization constants for cubic spline
    sigma_map = {1: 2/3, 2: 10/(7*np.pi), 3: 1/np.pi}
    sigma = sigma_map[dim] / (h**dim)
    # kernel definition support for q <= 2
    if q < 1.0:
        W = sigma * (1 - 1.5*q**2 + 0.75*q**3)
    elif q < 2.0:
        W = sigma * 0.25 * (2 - q)**3
    else:
        W = 0.0
    return W


def cubic_spline_grad(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of cubic spline kernel for 1D, 2D, or 3D.

    Parameters
    ----------
    r   : ndarray, shape (3,)  - displacement vector between two particles
    h   : float               - smoothing length
    dim : int                 - dimensionality (1, 2, or 3)

    Returns
    -------
    gradW : ndarray, shape (3,) - gradient vector
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
    r3 = np.asarray(r, dtype=float)
    if r3.shape != (3,):
        raise ValueError(f"Input r must have shape (3,), got {r3.shape}")
    # consider only first dim components
    r_vec = r3[:dim]
    r_norm = np.linalg.norm(r_vec)
    q = r_norm / h
    # zero gradient for r_norm=0 to avoid division by zero
    if r_norm == 0 or q > 2.0:
        return np.zeros(3, dtype=float)
    # normalization constants
    sigma_map = {1: 2/3, 2: 10/(7*np.pi), 3: 1/np.pi}
    sigma = sigma_map[dim] / (h**dim)
    # compute dW/dq
    if q < 1.0:
        dW_dq = sigma * (-3*q + 2.25*q**2)
    else:  # 1 <= q < 2.0
        dW_dq = sigma * (-0.75 * (2 - q)**2)
    # chain rule: dW/dr = dW/dq * (1/h)
    dW_dr = dW_dq / h
    # directional gradient
    grad = np.zeros(3, dtype=float)
    grad_dir = dW_dr * (r_vec / r_norm)
    grad[:dim] = grad_dir
    return grad
