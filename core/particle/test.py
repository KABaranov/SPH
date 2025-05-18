import numpy as np

from particle_dataclass import Particle
from get_param import get_particle_for_xyz


if __name__ == "__main__":
    # Конфигурация
    Nx, Ny, Nz = 10, 10, 1
    dim = 2
    H = 1.0

    # Итерационный процесс
    particles = []
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                particle = Particle(id=len(particles) + 1,
                                    m=0, rho=1000, p=0,
                                    x=np.array([x, y, z]),
                                    v=np.array([0, 0, 0]),
                                    h=H, neigh=[], neigh_w=[],
                                    drho_dt=0, dv_dt=np.array([0, 0, 0]),
                                    state=1)
                particles.append(particle)

    # Вывод
    for particle in particles:
        print(get_particle_for_xyz(particle))
