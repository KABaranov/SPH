import numpy as np


def gaussian_kernel(r: np.ndarray, h: float, dim: int, rcut: float = 3.0) -> float:
    """
    Truncated Gaussian kernel for 1D, 2D, or 3D using a 3-vector input.

    Parameters
    ----------
    r    : ndarray, shape (3,)  - displacement vector between two particles
    h    : float               - smoothing length
    dim  : int                 - dimensionality (1, 2, or 3)
    rcut : float               - support radius in units of h (default 3h)

    Returns
    -------
    W : float                  - kernel value
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
    r3 = np.asarray(r, dtype=float)
    if r3.shape != (3,):
        raise ValueError(f"Input r must have shape (3,), got {r3.shape}")
    # consider only first dim components
    r_vec = r3[:dim]
    r_norm = np.linalg.norm(r_vec)
    q = r_norm / h
    # apply cutoff
    if q >= rcut:
        return 0.0
    # normalization constant for Gaussian: sigma = 1/(pi^{dim/2} * h^dim)
    sigma = 1.0 / (np.pi**(dim / 2.0) * h**dim)
    W = sigma * np.exp(-q**2)
    return W


def gaussian_grad(r: np.ndarray, h: float, dim: int, rcut: float = 3.0) -> np.ndarray:
    """
    Gradient of truncated Gaussian kernel for 1D, 2D, or 3D.

    Parameters
    ----------
    r    : ndarray, shape (3,)  - displacement vector
    h    : float               - smoothing length
    dim  : int                 - dimensionality (1, 2, or 3)
    rcut : float               - support radius in units of h

    Returns
    -------
    gradW : ndarray, shape (3,) - gradient vector
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
    r3 = np.asarray(r, dtype=float)
    if r3.shape != (3,):
        raise ValueError(f"Input r must have shape (3,), got {r3.shape}")
    # consider only first dim components
    r_vec = r3[:dim]
    r_norm = np.linalg.norm(r_vec)
    q = r_norm / h
    # outside support or at zero distance
    if r_norm == 0.0 or q >= rcut:
        return np.zeros(3, dtype=float)
    # compute kernel value
    sigma = 1.0 / (np.pi**(dim / 2.0) * h**dim)
    W = sigma * np.exp(-q**2)
    # gradient magnitude: dW/dr = -2*q/h * W
    dW_dr = -2.0 * q * W / h
    # directional gradient
    grad = np.zeros(3, dtype=float)
    grad_dir = dW_dr * (r_vec / r_norm)
    grad[:dim] = grad_dir
    return grad
