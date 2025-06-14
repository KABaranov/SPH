import numpy as np
from typing import List, Sequence

from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.equations.compute_drho_dt import compute_drho_dt
from SPH.core.equations.compute_densities import compute_densities
from SPH.core.equations.eos import eos
from SPH.core.equations.compute_accelerations import compute_accelerations
from SPH.core.time_integrator.euler_cromer import euler_cromer
from SPH.core.pst.monaghan import apply_pst_monaghan
from SPH.core.pst.oger import apply_pst_oger


def step_sph(cfg: Config, particles: List[Particle], dt: float, box: Sequence[float] | float | None = None,):
    # 0. Обновить список соседей
    cfg.neighbor_search(particles, h=cfg.h, kernel=cfg.kernel, grad_kernel=cfg.grad, box=box, qmax=cfg.qmax)

    # # 1. Рассчитать dρ/dt (континуитет)
    # compute_drho_dt(particles=particles)
    #
    # # 2. Интегрировать плотность
    # for p in particles:
    #     p.rho += p.drho_dt * dt

    compute_densities(particles=particles)

    # 3. Рассчитать давление через уравнение состояния
    for p in particles:
        p.p = eos(rho=p.rho, cfg=cfg)

    # 4. Рассчитать силы: давление, вязкость, гравитацию…
    for i, p in enumerate(particles):
        # сбросим ускорение
        p.dv_dt = np.zeros_like(p.x, dtype=float)
    compute_accelerations(cfg=cfg, particles=particles)

    # 5. Интегрировать скорость и позицию (Euler–Cromer)
    euler_cromer(particles=particles, dt=dt)

    # apply_pst_monaghan(cfg=cfg, particles=particles)
    apply_pst_oger(cfg=cfg, particles=particles)

    # # 6. (опционально) Интегрировать температуру/энтальпию (теплопроводность)
    # if cfg.solve_heat:
    #     compute_dT_dt(cfg, particles)
    #     for p in particles:
    #         p.T += p.dT_dt * dt
