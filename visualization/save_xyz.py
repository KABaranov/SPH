from SPH.core.particle.particle_dataclass import Particle
from SPH.core.particle.get_param import get_particle_for_xyz
from typing import List, TextIO


def save_xyz(particles: List[Particle], f: TextIO) -> None:
    r"""Возвращает кадр xyz для частиц"""
    f.write(f"{len(particles)}\n\n")
    for particle in particles:
        f.write(get_particle_for_xyz(particle) + "\n")
