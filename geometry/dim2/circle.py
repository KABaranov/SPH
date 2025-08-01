import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Set


def generate_circle_points(
        center: Tuple[float, float],
        radius: float,
        dx: float
) -> Set[Tuple[float, float]]:
    """
    Генерирует точки внутри круга с заданным центром и радиусом.

    Параметры:
    ----------
    center : Tuple[float, float]
        Координаты центра круга (x, y).
    radius : float
        Радиус круга.
    dx : float
        Шаг между точками.

    Возвращает:
    -----------
    Set[Tuple[float, float]]
        Множество точек внутри круга.
    """
    cx, cy = center
    min_x, max_x = cx - radius, cx + radius
    min_y, max_y = cy - radius, cy + radius

    # Создаем сетку точек
    x_coords = np.arange(min_x, max_x + dx, dx)
    y_coords = np.arange(min_y, max_y + dx, dx)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Фильтруем точки, принадлежащие кругу
    circle_points = set()
    for x, y in zip(grid_x.ravel(), grid_y.ravel()):
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            circle_points.add((float(x), float(y)))

    return circle_points


def plot_circle(
        center: Tuple[float, float],
        radius: float,
        points: Set[Tuple[float, float]],
        title: str = "Круг с частицами"
):
    """Отрисовывает круг и точки."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Рисуем круг
    circle = plt.Circle(center, radius, fill=False, color='k', linewidth=2)
    ax.add_patch(circle)

    # Рисуем точки
    points_array = np.array(list(points))
    ax.scatter(points_array[:, 0], points_array[:, 1], s=10, color='blue', alpha=0.6, label="Частицы")

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{title} (dx={dx})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Пример использования
    center = (0.5, 0.5)
    radius = 1.0
    dx = 0.05

    points = generate_circle_points(center, radius, dx)

    # Отрисовка
    plot_circle(center, radius, points)

    # Вывод первых 10 точек для проверки
    print("Пример точек:", sorted(points)[:10])
