from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.equations.compute_densities import compute_densities

import numpy as np
import matplotlib.pyplot as plt


# Тест 1: Однородная решётка 2D
def density_test1(cfg: Config, out_plot=False) -> None:
    print("Проверка расчёта плотности (Тест 1):")
    for param in ["width", "height", "dx"]:
        if param not in cfg.scenario_param.keys():
            raise ValueError(f"Необходимо указать {param} в параметрах сценария (density_test1)")
    width, height, dx = cfg.scenario_param["width"], cfg.scenario_param["height"], cfg.scenario_param["dx"]
    qmax = 3 * cfg.h
    kernel = cfg.kernel
    x, y = [i * dx for i in range(int(width // dx) + 2)], [i * dx for i in range(int(height // dx) + 2)]
    particles = []
    for xi in x:
        for yi in y:
            p = Particle(
                id=len(particles), m=cfg.rho0*(dx**cfg.dim), p=0, x=np.array([xi, yi, 0]),
                drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=cfg.h,
                neigh=[], neigh_w=[], rho=cfg.rho0, v=np.array([0, 0, 0])
            )
            particles.append(p)

    # build_neigh(particles, h=cfg.h, dim=cfg.dim, box=-1, qmax=10.0, kernel=kernel)
    for pi in particles:
        for pj in particles:
            dr = pj.x - pi.x
            if np.linalg.norm(dr) <= qmax:
                pi.neigh.append(pj.id)
                pi.neigh_w.append(kernel(r=dr, h=pi.h, dim=cfg.dim))

    compute_densities(particles)

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
        line_colors = cmap(norm(rho_out))
        plt.scatter(x_out, y_out, color=line_colors)
        plt.show()
