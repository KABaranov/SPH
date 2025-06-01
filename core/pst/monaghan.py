import numpy as np
from typing import List
from sph.pst.base import IPST
from sph.core.particle import Particle


class MonaghanXSPH(IPST):
    """
    Классическая XSPH коррекция скоростей по Монaгану.
    δu_i = ε * Σ_j (m_j / ρ_j) (v_j - v_i) W_ij
    Скорость частиц обновляется v_i += δu_i.
    """
    def __init__(self, eps: float = 0.5):
        self.eps = eps

    def apply(self, particles: List[Particle], dt: float) -> None:
        # Сохраним оригинальные скорости, чтобы вычисления не искажались
        orig_vs = [p.v.copy() for p in particles]
        for i, pi in enumerate(particles):
            delta_u = np.zeros_like(pi.v)
            for j_idx, w in zip(pi.neigh, pi.W):
                pj = particles[j_idx]
                vij = orig_vs[j_idx] - orig_vs[i]
                rhoij = pi.rho + pj.rho
                delta_u += pj.m * vij / rhoij * w
            pi.v += 2.0 * self.eps * delta_u

    # def apply(self, particles: List[Particle], dt: float) -> None:
    #     # Сохраним оригинальные скорости, чтобы вычисления не искажались
    #     orig_vs = [p.v.copy() for p in particles]
    #     for i, pi in enumerate(particles):
    #         delta_u = np.zeros_like(pi.v)
    #         for j_idx, w in zip(pi.neigh, pi.W):
    #             pj = particles[j_idx]
    #             vij = orig_vs[j_idx] - orig_vs[i]
    #             delta_u += pj.m / pj.rho * vij * w
    #         pi.v += self.eps * delta_u
