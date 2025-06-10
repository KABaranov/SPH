from SPH.core.particle.particle_dataclass import Particle

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable
from time import time

from SPH.configs.get_config import get_config
from SPH.core.neighbor_search.bruteforce import bruteforce
from SPH.core.neighbor_search.cell_linked import cell_linked
from SPH.core.neighbor_search.kd_tree import kd_tree
from SPH.core.kernels.wendland_c2 import wendland_c2_kernel


def ns_multitest_1d(functions: dict,
                    is_periodic: bool = False) -> None:
    dx_list = [0.1 * (1/2)**i for i in range(8)]
    Lx = 100
    dim = 1
    rho0 = 1000
    kappa0 = 3.0
    qmax = 1.9
    for name in functions.keys():
        search_function = functions[name](dim)
        result = []
        n_part = []
        for dxi in dx_list:
            particles0 = []
            dx = dxi
            h = dx * kappa0
            x = [i * dx for i in range(int(Lx // dx) + 2)]
            for xi in x:
                p = Particle(
                    id=len(particles0), m=rho0 * (dx ** dim), p=0, x=np.array([xi, 0, 0]),
                    drho_dt=0, dv_dt=np.array([0, 0, 0]), state=1, h=h,
                    neigh=[], neigh_w=[], rho=rho0, v=np.array([0, 0, 0])
                )
                particles0.append(p)
            n_part.append(len(particles0))
            start_time = time()
            search_function(particles=particles0, h=h, kernel=wendland_c2_kernel, box=None, qmax=qmax)
            result_time = time() - start_time
            result.append(result_time)
            print(f"Время поиска {name} на dx: {dxi}: {result_time}")
        plt.plot(n_part, result)
    plt.title("Сравнение методов поиска соседей")
    plt.xlabel("Количество частиц")
    plt.ylabel("Время поиска (с)")
    plt.legend(functions.keys())
    plt.show()


if __name__ == "__main__":
    neigh_func = {
        "Brute force": bruteforce,
        "Cell linked": cell_linked,
        "KD-Tree": kd_tree
    }
    ns_multitest_1d(neigh_func, is_periodic=False)
