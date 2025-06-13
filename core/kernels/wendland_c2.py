import numpy as np


def wendland_c2_kernel(r: np.ndarray, h: float, dim: int) -> float:
    """
    Wendland C2 kernel for 1D, 2D, or 3D, always accepts r as length-3 vector.

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
    if q >= 2.0:
        return 0.0
    # normalization constants for Wendland C2
    sigma_map = {1: 3/4, 2: 7/(4*np.pi), 3: 21/(16*np.pi)}
    sigma = sigma_map[dim] / (h**dim)
    u = 1.0 - 0.5*q
    phi = u**4 * (2.0*q + 1.0)
    return sigma * phi


def wendland_c2_grad(r: np.ndarray, h: float, dim: int) -> np.ndarray:
    """
    Gradient of Wendland C2 kernel for 1D, 2D, or 3D.

    Parameters
    ----------
    r   : ndarray, shape (3,)  - displacement vector
    h   : float                - smoothing length
    dim : int                  - dimensionality (1, 2, or 3)

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
    # outside support or zero distance
    if r_norm == 0.0 or q >= 2.0:
        return np.zeros(3, dtype=float)
    # normalization
    sigma_map = {1: 3/4, 2: 7/(4*np.pi), 3: 21/(16*np.pi)}
    sigma = sigma_map[dim] / (h**dim)
    # radial derivative dphi/dq = -5*q*(1 - 0.5*q)**3
    u = 1.0 - 0.5*q
    dphi_dq = -5.0 * q * (u**3)
    # dW/dr = sigma * (dphi/dq) * (1/h)
    dW_dr = sigma * dphi_dq / h
    # vector gradient: pad to length 3
    grad = np.zeros(3, dtype=float)
    grad_dir = (dW_dr * (r_vec / r_norm))
    grad[:dim] = grad_dir
    return grad
