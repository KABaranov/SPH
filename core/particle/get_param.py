from SPH.core.particle.particle_dataclass import Particle

import numpy as np
from typing import List


def get_id(particle: Particle) -> int:
    return particle.id


def get_m(particle: Particle) -> float:
    return particle.m


def get_rho(particle: Particle) -> float:
    return particle.rho


def get_p(particle: Particle) -> float:
    return particle.p


def get_xyz(particle: Particle) -> np.ndarray:
    return particle.x


def get_x(particle: Particle) -> float:
    return particle.x[0]


def get_y(particle: Particle) -> float:
    return particle.x[1]


def get_z(particle: Particle) -> float:
    return particle.x[2]


def get_v(particle: Particle) -> np.ndarray:
    return particle.v


def get_vx(particle: Particle) -> float:
    return particle.v[0]


def get_vy(particle: Particle) -> float:
    return particle.v[1]


def get_vz(particle: Particle) -> float:
    return particle.v[2]


def get_h(particle: Particle) -> float:
    return particle.h


def get_neigh(particle: Particle) -> List[int]:
    return particle.neigh


def get_neigh_w(particle: Particle) -> List[float]:
    return particle.neigh_w


def get_drho_dt(particle: Particle) -> float:
    return particle.drho_dt


def get_dv_dt(particle: Particle) -> np.ndarray:
    return particle.dv_dt


def get_dvx_dt(particle: Particle) -> float:
    return particle.dv_dt[0]


def get_dvy_dt(particle: Particle) -> float:
    return particle.dv_dt[1]


def get_dvz_dt(particle: Particle) -> float:
    return particle.dv_dt[2]


def get_state(particle: Particle) -> int:
    return particle.state


def get_T(particle: Particle) -> float:
    return particle.T


def get_k(particle: Particle) -> float:
    return particle.k


def get_c(particle: Particle) -> float:
    return particle.c


def get_particle_for_xyz(particle: Particle) -> str:
    r"""

    :param particle: Частица, параметры которой возвращаем
    :return:
        Строка для файла *.xyz: id, x, y, z, vx, vy, vz, m, rho, p,
        h, drho_dt, dvx_dt, dvy_dt, dvz_dt, state, T, k, c
    """
    return f"{particle.id} " \
           f"{particle.x[0]} {particle.x[1]} {particle.x[2]} " \
           f"{particle.v[0]} {particle.v[1]} {particle.v[2]} " \
           f"{particle.m} {particle.rho} {particle.p} " \
           f"{particle.h} {particle.drho_dt} " \
           f"{particle.dv_dt[0]} {particle.dv_dt[1]} {particle.dv_dt[2]} {particle.state} " \
           f"{particle.T} {particle.k} {particle.c}"
