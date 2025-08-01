import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Union


def point_in_triangle(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    """Проверяет, находится ли точка pt внутри треугольника p1-p2-p3."""
    v0 = p3 - p1
    v1 = p2 - p1
    v2 = pt - p1

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


def generate_quadrilateral_points(
        points: List[Union[Tuple[float, float], List[float], np.ndarray]],
        dx: float,
) -> Set[Tuple[float, float]]:
    """
    Генерирует точки внутри четырёхугольника с шагом dx.

    Параметры:
    ----------
    points : List[Point]
        Список из 4 точек четырёхугольника в порядке обхода (по или против часовой стрелки).
    dx : float
        Шаг между точками.

    Возвращает:
    -----------
    Set[Tuple[float, float]]
        Множество точек внутри четырёхугольника.
    """
    if len(points) != 4:
        raise ValueError("Четырёхугольник должен иметь 4 точки.")

    p1, p2, p3, p4 = [np.array(p, dtype=float) for p in points]
    px, py = [p[0] for p in points], [p[1] for p in points]

    min_x, max_x = min(px), max(px)
    min_y, max_y = min(py), max(py)

    # Создаем сетку точек и преобразуем в множество кортежей
    x_coords = np.arange(min_x, max_x + dx, dx)
    y_coords = np.arange(min_y, max_y + dx, dx)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    all_points = {(float(x), float(y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())}

    # Разбиваем четырёхугольник на два треугольника и проверяем точки
    quad_points = set()
    for (x, y) in all_points:
        pt = np.array([x, y])
        if point_in_triangle(pt, p1, p2, p3) or point_in_triangle(pt, p1, p3, p4):
            quad_points.add((x, y))

    return quad_points


if __name__ == "__main__":
    # Пример использования
    p1, p2, p3, p4 = (0.9, 0.7), (0.6, -0.9), (-0.6, 0.1), (-0.3, 1.2)
    dx = 0.1
    points = generate_quadrilateral_points([p1, p2, p3, p4], dx)

    # Отрисовка
    fig, ax = plt.subplots(figsize=(8, 6))

    # Рисуем четырёхугольник
    quad_line = np.vstack([p1, p2, p3, p4, p1])  # Замыкаем
    ax.plot(quad_line[:, 0], quad_line[:, 1], 'k-', linewidth=2, label="Четырёхугольник")

    # Рисуем точки (преобразуем множество в массив для scatter)
    points_array = np.array(list(points))
    ax.scatter(points_array[:, 0], points_array[:, 1], s=10, color='blue', alpha=0.6, label="Частицы")

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Четырёхугольник с частицами (dx={dx})")
    plt.tight_layout()
    plt.show()

    # Вывод первых 10 точек для проверки
    print("Пример точек:", list(points)[:10])