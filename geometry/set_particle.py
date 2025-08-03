from SPH.core.particle.particle_dataclass import Particle
import numpy as np
from typing import Set, Tuple, List


def set_particle(positions: Set[Tuple[float, float, float]], id0: int,
                 rho0: float, dx: float, dim: int, p: float, h: float,
                 drho_dt: float, state: int, T: float, k: float,
                 c: float) -> List[Particle]:
    particles = []
    for pos in positions:
        p = Particle(
            id=len(particles) + id0, m=rho0 * (dx ** dim), p=p, x=np.array([pos[0], pos[1], pos[2]]),
            drho_dt=drho_dt, dv_dt=np.array([0, 0, 0]), state=state, h=h, neigh=[],
            neigh_w=[], grad_w=[], rho=rho0, v=np.zeros(3), T=T, k=k, c=c
        )
        particles.append(p)
    return particles
