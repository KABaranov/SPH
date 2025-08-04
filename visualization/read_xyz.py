import numpy as np
from typing import List, Tuple, Optional, Dict


def read_xyz_for_visualization(
        filename: str,
        parameter: str = "rho",
        parameter_column: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Читает весь .xyz файл и возвращает список кадров с координатами и параметрами.

    Args:
        filename: Путь к файлу .xyz
        parameter: Имя параметра для извлечения ("rho", "p", "T" и т.д.)
        parameter_column: Номер колонки параметра (если нужно указать вручную)

    Returns:
        Список кортежей (coords, values), где:
        - coords: массив координат (N, 3)
        - values: массив значений параметра (N,)
    """
    # Соответствие имени параметра номеру колонки
    param_cols = {
        "id": 0, "x": 1, "y": 2, "z": 3,
        "vx": 4, "vy": 5, "vz": 6,
        "m": 7, "rho": 8, "p": 9,
        "h": 10, "drho_dt": 11,
        "dvx_dt": 12, "dvy_dt": 13, "dvz_dt": 14,
        "state": 15, "T": 16, "k": 17, "c": 18,
    }

    if parameter_column is None:
        if parameter not in param_cols:
            raise ValueError(f"Unknown parameter: {parameter}. Available: {list(param_cols.keys())}")
        col = param_cols[parameter]
    else:
        col = parameter_column

    frames = []

    with open(filename, 'r') as f:
        while True:
            # Читаем количество частиц
            line = f.readline()
            if not line:  # Конец файла
                break
            n_particles = int(line.strip())

            # Пропускаем пустую строку
            f.readline()

            # Читаем данные частиц
            coords = []
            values = []
            for _ in range(n_particles):
                parts = f.readline().strip().split()
                if not parts:
                    continue

                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                values.append(float(parts[col]))

            frames.append((np.array(coords), np.array(values)))

    return frames
