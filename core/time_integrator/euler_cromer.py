from typing import List

from SPH.core.particle.particle_dataclass import Particle


def euler_cromer(particles: List[Particle], dt: float):
    for pi in particles:
        pi.v = pi.v + pi.dv_dt * dt
        pi.x = pi.x + pi.v * dt
