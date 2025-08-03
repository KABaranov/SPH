from SPH.configs.config_class import Config
from SPH.core.equations.compute_densities import compute_densities
from SPH.core.step_sph import step_sph
from SPH.geometry.set_particle import set_particle
from SPH.geometry.dim2.rectangle import generate_rectangle_points
from SPH.visualization.save_xyz import save_xyz
from SPH.visualization.create_xyz_file import create_xyz_file

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm


# ======================================
# Квадратная капля
# ======================================
def square_drop(cfg: Config) -> None:
    # Параметры сценария
    width = cfg.scenario_param['width']
    height = cfg.scenario_param['height']
    dx = cfg.scenario_param['dx']
    iterations = int(cfg.total_time / cfg.dt)
    box = (width, height) if cfg.is_periodic else None
    xyz_file = create_xyz_file(cfg.scenario_name)

    # Инициализация частиц
    particles = []
    positions = generate_rectangle_points(
        points=[(0, 0), (0, height), (width, height), (width, 0)], dx=dx
    )
    particles += (set_particle(positions=positions, id0=len(particles), rho0=cfg.rho0, dx=dx, dim=cfg.dim,
                               p=0, h=cfg.h, drho_dt=0, state=1, T=0, k=0, c=0))

    # Запускаем симуляцию и сохраняем данные для каждого кадра
    frames = []  # list of (x, y, rho)
    cfg.neighbor_search(particles=particles, h=cfg.h, kernel=cfg.kernel, grad_kernel=cfg.grad, box=box, qmax=cfg.qmax)
    compute_densities(particles=particles)
    xs = np.array([p.x[0] for p in particles])
    ys = np.array([p.x[1] for p in particles])
    rhos = np.array([p.rho for p in particles])
    frames.append((xs, ys, rhos))

    for step in tqdm(range(iterations)):
        if cfg.corrector and step % cfg.corrector_period == 0:
            cfg.corrector(particles=particles, n_iter=cfg.corrector_iter)
        step_sph(cfg, particles, cfg.dt, box=box, step=step)
        xs = np.array([p.x[0] for p in particles])
        ys = np.array([p.x[1] for p in particles])
        rhos = np.array([p.rho for p in particles])
        frames.append((xs, ys, rhos))
        save_xyz(particles=particles, f=xyz_file)
    xyz_file.close()

    # Границы плотности
    all_rhos = np.hstack([f[2] for f in frames])
    rho_min, rho_max = all_rhos.min(), all_rhos.max()

    # Рисуем первый кадр
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.subplots_adjust(bottom=0.2)
    sc = ax.scatter(frames[-1][0], frames[-1][1], c=frames[-1][2], cmap='viridis',
                    vmin=rho_min, vmax=rho_max)
    ax.set_title('Step 0')
    plt.colorbar(sc, ax=ax, label='rho')

    # Добавляем слайдер
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(ax_slider, 'Step', 0, iterations - 1, valinit=iterations, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        xs, ys, rhos = frames[idx]
        sc.set_offsets(np.c_[xs, ys])
        sc.set_array(rhos)
        ax.set_title(f'Step {idx}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
