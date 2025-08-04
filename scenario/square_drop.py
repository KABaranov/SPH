from SPH.configs.config_class import Config
from SPH.core.equations.compute_densities import compute_densities
from SPH.core.step_sph import step_sph
from SPH.geometry.set_particle import set_particle
from SPH.geometry.dim2.rectangle import generate_rectangle_points
from SPH.visualization.save_xyz import save_xyz
from SPH.visualization.create_xyz_file import create_xyz_file

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

    # Запускаем симуляцию
    cfg.neighbor_search(particles=particles, h=cfg.h, kernel=cfg.kernel, grad_kernel=cfg.grad, box=box, qmax=cfg.qmax)
    compute_densities(particles=particles)

    for step in tqdm(range(iterations)):
        if cfg.corrector and step % cfg.corrector_period == 0:
            cfg.corrector(particles=particles, n_iter=cfg.corrector_iter)
        step_sph(cfg, particles, cfg.dt, box=box, step=step)
        save_xyz(particles=particles, f=xyz_file)
    xyz_file.close()
