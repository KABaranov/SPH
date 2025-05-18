from particle_dataclass import Particle
import numpy as np
from typing import List, Optional


def set_id(particle: Particle, idx: int) -> None:
    particle.id = idx


def set_m(particle: Particle, m: float) -> None:
    particle.m = m


def set_rho(particle: Particle, rho: float) -> None:
    particle.rho = rho


def set_p(particle: Particle, p: float) -> None:
    particle.p = p


def set_xyz(particle: Particle, xyz: np.ndarray) -> None:
    particle.x = xyz


def set_x(particle: Particle, x: float) -> None:
    particle.x[0] = x


def set_y(particle: Particle, y: float) -> None:
    particle.x[1] = y


def set_z(particle: Particle, z: float) -> None:
    particle.x[2] = z


def set_v(particle: Particle, vel: np.ndarray) -> None:
    particle.v = vel


def set_vx(particle: Particle, vel_x: float) -> None:
    particle.v[0] = vel_x


def set_vy(particle: Particle, vel_y: float) -> None:
    particle.v[1] = vel_y


def set_vz(particle: Particle, vel_z: float) -> None:
    particle.v[2] = vel_z


def set_h(particle: Particle, h: float) -> None:
    particle.h = h


def set_neigh(particle: Particle, neigh: List[int]) -> None:
    particle.neigh = neigh


def set_neigh_w(particle: Particle, neigh_w: List[float]) -> None:
    particle.neigh_w = neigh_w


def set_drho_dt(particle: Particle, drho_dt: float) -> None:
    particle.drho_dt = drho_dt


def set_dv_dt(particle: Particle, dv_dt: np.ndarray) -> None:
    particle.dv_dt = dv_dt


def set_dvx_dt(particle: Particle, dvx_dt: float) -> None:
    particle.dv_dt[0] = dvx_dt


def set_dvy_dt(particle: Particle, dvy_dt: float) -> None:
    particle.dv_dt[1] = dvy_dt


def set_dvz_dt(particle: Particle, dvz_dt: float) -> None:
    particle.dv_dt[2] = dvz_dt


def set_state(particle: Particle, state: int) -> None:
    particle.state = state


def set_particle_param(particle: Particle, idx: Optional[int], m: Optional[float], rho: Optional[float],
                       p: Optional[float], x: Optional[np.ndarray], v: Optional[np.ndarray], h: Optional[float],
                       neigh: Optional[List[int]], neigh_w: Optional[List[float]], drho_dt: Optional[float],
                       dv_dt: Optional[np.ndarray], state: Optional[int]) -> None:
    if idx:
        particle.id = idx
    if m:
        particle.m = m
    if rho:
        particle.rho = rho
    if p:
        particle.p = p
    if x:
        particle.x = x
    if v:
        particle.v = v
    if h:
        particle.h = h
    if neigh:
        particle.neigh = neigh
    if neigh_w:
        particle.neigh_w = neigh_w
    if drho_dt:
        particle.drho_dt = drho_dt
    if dv_dt:
        particle.dv_dt = dv_dt
    if state:
        particle.state = state
