from typing import List
from ..particle.particle_dataclass import Particle
from .eos import eos
from SPH.configs.config_class import Config


# ------------------------------------------------------------------
# Обновление давления
# ------------------------------------------------------------------
def compute_pressure(cfg: Config, particles: List[Particle]) -> None:
    for p in particles:
        p.p = eos(p.rho, cfg)
