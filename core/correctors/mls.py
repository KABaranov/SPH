def mls_correction(particles: List[Particle], L: float = -1.0, n_iter: int = 2) -> None:
    """
    L - длина рассматриваемой зоны (для периодических границ
    Однопроходная MLS-коррекция плотности:
    ρ_i ← Σ m_j W_ij / ( a_0 + aᵀ·0 )   (нулевой и линейный порядки).
    Корректирует нулевой и первый моменты ⇒ устраняет O(Δx) остаток.
    Работает при любых массах.
    """
    for _ in range(n_iter):
        rho0 = [p.rho for p in particles]  # исходная плотность
        for i, pi in enumerate(particles):
            M0 = M1 = M2 = 0.0
            for w, j in zip(pi.neigh_w, pi.neigh):
                pj = particles[j]
                dx = pj.x[0] - pi.x[0]
                if L != -1:
                    dx = dx_periodic(dx, L)
                fac = pj.m / rho0[j]  # ОБЯЗАТЕЛЬНО ρ₀!
                M0 += fac * w
                M1 += fac * w * dx
                M2 += fac * w * dx * dx
            a = -M1 / M2  # коэффициент линейной поправки

            num = 0.0
            for w, j in zip(pi.neigh_w, pi.neigh):
                pj = particles[j]
                dx = pj.x[0] - pi.x[0]
                if L != -1:
                    dx = dx_periodic(dx, L)
                num += particles[j].m * w * (1.0 + a * dx)  # ← линейный множитель

            pi.rho = num / (M0 + a * M1)  # = Σ (m/ρ₀) W (1+a·dx)