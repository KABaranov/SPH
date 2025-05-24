"""sph/core/equations.py
---------------------------------
Уравнения ГСЧ (SPH) в компактном классе `SPHEquations`.

Особенности реализации
======================
* **Уравнение состояния Тейта** с «флорой» (p ≥ 0), чтобы подавить
  тензиональную нестабильность.
* **Искусственная вязкость** Монaгана (α, β, ε).
* Раздельные методы:
  - `compute_densities`  — сумма ядер,
  - `compute_pressure`   — EOS,
  - `compute_drho_dt`    — континуитет,
  - `compute_accelerations` — градиент давления + вязкость + внешняя сила.
* Поддержка 2D и 3D (определяется размерностью координат `Particle.x`).
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from core.particle import Particle
from core.kernels import Kernel


class SPHEquations:
    """Набор уравнений для SPH‑расчёта."""

    def __init__(
        self,
        rho0: float,
        gamma: float = 7.0,
        B: Optional[float] = None,
        c0: Optional[float] = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        epsilon: float = 0.01,
        p_floor: float = 0.0,  # отсечка отрицат. давлений
    ) -> None:
        self.rho0 = rho0
        self.gamma = gamma
        if B is None:
            if c0 is None:
                raise ValueError("Нужно указать B или c0 для EOS")
            self.B = rho0 * c0**2 / gamma
            self.c0 = c0
        else:
            self.B = B
            self.c0 = np.sqrt(B * gamma / rho0)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.p_floor = p_floor

    # ------------------------------------------------------------------
    # Уравнение состояния
    # ------------------------------------------------------------------
    def eos(self, rho: float) -> float:
        """Tait EOS: p = B * ((rho / rho0)^gamma − 1)  c отсечкой p ≥ p_floor"""
        p = self.B * ((rho / self.rho0) ** self.gamma - 1.0)
        return max(p, self.p_floor)

    # ------------------------------------------------------------------
    # Расчёт плотности
    # ------------------------------------------------------------------
    def compute_densities(self, particles: List[Particle]) -> None:
        """ρᵢ = Σⱼ mⱼ W(rᵢⱼ)"""
        for pi in particles:
            rho = 0.0
            for w, j in zip(pi.W, pi.neigh):
                rho += particles[j].m * w
            pi.rho = rho

    # ------------------------------------------------------------------
    # Обновление давления
    # ------------------------------------------------------------------
    def compute_pressure(self, particles: List[Particle]) -> None:
        for p in particles:
            p.p = self.eos(p.rho)

    # ------------------------------------------------------------------
    # Континуитет: dρ/dt
    # ------------------------------------------------------------------
    def compute_drho_dt(self, particles: List[Particle], kernel: Kernel) -> None:
        for i, pi in enumerate(particles):
            drho = 0.0
            for j in pi.neigh:
                pj = particles[j]
                vij = pi.v - pj.v
                grad = kernel.grad_W(pi.x - pj.x)
                drho += pj.m * np.dot(vij, grad)
            pi.drho_dt = drho

    # ------------------------------------------------------------------
    # Уравнение движения
    # ------------------------------------------------------------------
    def compute_accelerations(
        self,
        particles: List[Particle],
        kernel: Kernel,
        external_force: Optional[np.ndarray] = None,
    ) -> None:
        if external_force is None:
            external_force = np.zeros_like(particles[0].v)

        for i, pi in enumerate(particles):
            acc = np.zeros_like(pi.v)
            for j in pi.neigh:
                pj = particles[j]
                rij = pi.x - pj.x
                grad_w = kernel.grad_W(rij)

                # Давление (symmetric form)
                pij_term = pi.p / (pi.rho ** 2) + pj.p / (pj.rho ** 2)

                # Искусственная вязкость
                visc = self._artificial_viscosity(pi, pj, rij, kernel)

                acc -= pj.m * (pij_term + visc) * grad_w

            pi.dv_dt = acc + external_force

    # ------------------------------------------------------------------
    # Искусственная вязкость Монaгана
    # ------------------------------------------------------------------
    def _artificial_viscosity(
        self,
        pi: Particle,
        pj: Particle,
        rij: np.ndarray,
        kernel: Kernel,
    ) -> float:
        vij = pi.v - pj.v
        rij_dot = np.dot(vij, rij)
        if rij_dot >= 0:
            return 0.0  # расходятся — без вязкости

        h = kernel.h
        mu_ij = h * rij_dot / (np.dot(rij, rij) + self.epsilon * h ** 2)
        c_bar = self.c0  # для простоты: const звук
        rho_bar = 0.5 * (pi.rho + pj.rho)
        return (-self.alpha * c_bar * mu_ij + self.beta * mu_ij**2) / rho_bar
