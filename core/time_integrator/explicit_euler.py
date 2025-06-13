from typing import List

from SPH.core.particle.particle_dataclass import Particle
from SPH.configs.config_class import Config


def explicit_euler(particles: List[Particle], dt: float):
    for pi in particles:
        pi.x = pi.x + pi.v * dt
        pi.v = pi.v + pi.dv_dt * dt
