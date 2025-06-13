from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.equations.compute_densities import compute_densities
from SPH.core.step_sph import step_sph

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ======================================
# Квадратная капля
# ======================================
def square_drop(cfg: Config) -> None:
    print("Квадратная капля:")
    iterations = int(cfg.total_time // cfg.dt)
    out_plot = cfg.out_plot
    for param in ["width", "height", "dx"]:
        if param not in cfg.scenario_param.keys():
            raise ValueError(f"Необходимо указать {param} в параметрах сценария (square_drop)")
    width, height, dx = cfg.scenario_param["width"], cfg.scenario_param["height"], cfg.scenario_param["dx"]
    h, dim = cfg.h, cfg.dim
    box = (width, height) if cfg.is_periodic else None

    x, y = [i * dx for i in range(int(width // dx) + 1)], [i * dx for i in range(int(width // dx) + 1)]
    particles = []
    for xi in x:
        for yi in y:
            p = Particle(
                id=len(particles), m=cfg.rho0 * (dx ** dim), p=0, x=np.array([xi, yi, 0]),
                drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=cfg.h, neigh=[],
                grad_w=[], neigh_w=[], rho=cfg.rho0, v=np.array([0, 0, 0])
            )
            particles.append(p)

    for time in tqdm(range(iterations)):
        step_sph(cfg=cfg, particles=particles, dt=cfg.dt, box=box)

    x_out, y_out, rho_out = [], [], []
    for pi in particles:
        x_out.append(pi.x[0])
        y_out.append(pi.x[1])
        rho_out.append(pi.rho)
    rho_min, rho_max = min(rho_out), max(rho_out)
    print(f"\tМинимальная плотность: {rho_min}\n\tМаксимальная плотность {rho_max}")
    if out_plot:
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(rho_min, rho_max)
        # norm = plt.Normalize(900, 1000)
        line_colors = cmap(norm(rho_out))
        plt.scatter(x_out, y_out, color=line_colors)

        plt.show()
