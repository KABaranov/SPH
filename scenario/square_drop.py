from SPH.configs.config_class import Config
from SPH.core.particle.particle_dataclass import Particle
from SPH.core.equations.compute_densities import compute_densities
from SPH.core.step_sph import step_sph

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

    # Инициализация частиц
    x_vals = np.arange(0, width + dx, dx)
    y_vals = np.arange(0, height + dx, dx)
    particles = []
    for xi in x_vals:
        for yi in y_vals:
            # if xi in [x_vals[0], x_vals[-1]] and yi in [y_vals[0], y_vals[-1]]:
            #     continue
            p = Particle(
                id=len(particles), m=cfg.rho0 * dx ** cfg.dim, p=0,
                x=np.array([xi, yi, 0.]), drho_dt=0,
                dv_dt=np.zeros(3), state=1, h=cfg.h,
                neigh=[], grad_w=[], neigh_w=[], rho=cfg.rho0, v=np.zeros(3)
            )
            particles.append(p)

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
        step_sph(cfg, particles, cfg.dt, box=box)
        xs = np.array([p.x[0] for p in particles])
        ys = np.array([p.x[1] for p in particles])
        rhos = np.array([p.rho for p in particles])
        frames.append((xs, ys, rhos))

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
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
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
