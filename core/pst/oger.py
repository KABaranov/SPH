import numpy as np
from typing import List
from sph.pst.base import IPST
from sph.core.particle import Particle
from sph.core.kernels import Kernel


class OgerPST(IPST):
    """
    ALE-версия техники сдвига частиц по Огеру.
    δu_i = -β * h^2 * ∇C_i, где ∇C_i = Σ_j ∇W_ij
    Коррекция скорости: v_i += δu_i

    Args:
        beta: параметр PST Огера
        kernel: экземпляр SPH-ядер для вычисления градиента
    """
    def __init__(self, beta: float = 0.01, kernel: Kernel = None):
        if kernel is None:
            raise ValueError("OgerPST требует указать ядро для вычисления ∇W")
        self.beta = beta
        self.kernel = kernel

    def apply(self, particles: List[Particle], dt: float) -> None:
        # Вычисляем и применяем смещение для каждой частицы
        h2 = self.kernel.h ** 2
        for pi in particles:
            gradC = np.zeros_like(pi.v)
            # ∇C = Σ_j ∇W(x_i - x_j)
            for j_idx in pi.neigh:
                pj = particles[j_idx]
                gradC += self.kernel.grad_W(pi.x - pj.x)
            delta_u = - self.beta * h2 * gradC
            pi.v += delta_u
