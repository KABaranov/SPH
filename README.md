# SPH

## Структура проекта

Предыдущий вариант структуры (почти такой же, но не актуален)
```
SPH/                    # pip‑устанавливаемый пакет
│
├─ core/                # базовая «физика» ГСЧ, НЕ меняется при смене ТСЧ
│   ├─ particle         # @dataclass Particle
│   ├─ kernels          # CubicSpline, Wendland C², ...  
│   ├─ neighbor_search  # связка PyKD‑tree / cell‑linked list
│   ├─ equations        # ∂ρ/∂t, ∂v/∂t, EOS, без δu
│   ├─ time_integrator  # LeapFrog / PC / RK2  + CFL‑контроль
│   └─ diagnostics      # RMS‑ошибка ρ, анизотропия ζ, ΔV, Энергия
│
├─ pst/                 # семейство модулей смещения (ТСЧ)
│   ├─ __init__.py
│   ├─ base.py          # абстрактный класс ITSC
│   ├─ none.py          # δu ≡ 0
│   ├─ monaghan.py      # классическая XSPH‑формула
│   └─ oger.py          # ALE‑смещение по Огеру
│
├─ scenario/            # эталонные тест‑кейсы
│   ├─ shock_tube.py
│   ├─ lid_driven_cavity.py
│   ├─ dam_break.py
│   └─ sloshing_3d.py
│
├─ configs/             # YAML‑файлы параметров (ядро, h, ε, β, Δt…)
│   ├── common.yml           # ρ₀, c₀, γ, g, характеристические длины
├── ├── profiles/
│   └─  ├── water.yml        # профиль «вода»: Re=… , ν=… , σ=…
│       └── oil.yml          # профиль «масло»: Re=… , ν=… , σ=…
│   ├── dam_break.yml
│   ├── lid_driven_cavity.yml
│   …
│
├─ scripts/
│   ├─ run.py           # CLI‑драйвер: --scenario lid --tsc monaghan
│   ├─ sweep.py         # пакетный прогон всех комбинаций
│   └─ postprocess.ipynb# Jupyter‑анализ CSV → графики, таблицы
│
├─ pyproject.toml       # Poetry / PEP‑518 метаданные
└─ README.md

```