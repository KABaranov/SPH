from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.equations.compute_densities import compute_densities
from SPH.geometry.set_particle import set_particle
from SPH.geometry.dim2.rectangle import generate_rectangle_points

import numpy as np
import matplotlib.pyplot as plt


# Тест 1: Однородная решётка 2D
def density_test1(cfg: Config) -> None:
    out_plot = cfg.out_plot
    neighbor_search = cfg.neighbor_search
    print("Проверка расчёта плотности (Тест 1):")
    for param in ["width", "height", "dx"]:
        if param not in cfg.scenario_param.keys():
            raise ValueError(f"Необходимо указать {param} в параметрах сценария (density_test1)")
    width, height, dx = cfg.scenario_param["width"], cfg.scenario_param["height"], cfg.scenario_param["dx"]
    h, dim = cfg.h, cfg.dim
    if cfg.is_periodic:
        box = (width, height)
    else:
        box = None
    qmax = cfg.qmax
    kernel = cfg.kernel
    grad_kernel = cfg.grad

    # Инициализация частиц
    particles = []
    positions = generate_rectangle_points(
        points=[(0, 0), (0, height), (width, height), (width, 0)], dx=dx
    )
    particles += (set_particle(positions=positions, id0=len(particles), rho0=cfg.rho0, dx=dx, dim=cfg.dim,
                               p=0, h=cfg.h, drho_dt=0, state=1, T=0, k=0, c=0))

    neighbor_search(particles, h=h, box=box, qmax=qmax, kernel=kernel, grad_kernel=grad_kernel)

    compute_densities(particles)
    print(cfg.corrector_name)
    if cfg.corrector_name.lower() != "none":
        cfg.corrector(particles, cfg.corrector_iter)

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
