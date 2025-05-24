from typing import List
from ..particle.particle_dataclass import Particle
from eos import eos


# ------------------------------------------------------------------
# Обновление давления
# ------------------------------------------------------------------
def compute_pressure(particles: List[Particle]) -> None:
    for p in particles:
        p.p = eos(p.rho)
