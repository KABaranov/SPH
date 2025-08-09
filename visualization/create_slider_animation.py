from SPH.visualization.read_xyz import read_xyz_for_visualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Optional, Union, Tuple
from pathlib import Path


def create_slider_animation(
        data_source: Union[str, List[Tuple[np.ndarray, np.ndarray]]],
        parameter: str = "rho",
        parameter_column: Optional[int] = None,
        cmap: str = "viridis",
        aspect: str = "equal",
        figsize: tuple = None,
        title: str = "Particle Simulation",
) -> None:
    """
    Создает анимацию со слайдером.

    Args:
        data_source: Путь к .xyz файлу или список кадров
        parameter: Имя параметра для визуализации
        parameter_column: Номер колонки параметра
        cmap: Цветовая карта
        aspect: Соотношение осей
        figsize: Размер фигуры
        title: Заголовок
    """
    # Получаем кадры
    if isinstance(data_source, str):
        frames = read_xyz_for_visualization(data_source, parameter, parameter_column)
    elif isinstance(data_source, list):
        frames = data_source
    else:
        raise TypeError("data_source должен быть строкой или списком кадров")

    if not frames:
        raise ValueError("Нет данных для визуализации")

    # Подготовка данных
    all_values = np.hstack([f[1] for f in frames])
    vmin, vmax = all_values.min(), all_values.max()
    n_frames = len(frames)

    # Создание графика
    fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    ax.set_aspect(aspect)
    plt.subplots_adjust(bottom=0.2)

    # Первый кадр
    sc = ax.scatter(
        frames[-1][0][:, 0],  # x
        frames[-1][0][:, 1],  # y
        c=frames[-1][1],  # значения параметра
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"{title} (Кадр {n_frames - 1})")
    plt.colorbar(sc, ax=ax, label=parameter)

    # Слайдер
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(
        ax=ax_slider,
        label='Кадр',
        valmin=0,
        valmax=n_frames - 1,
        valinit=n_frames - 1,
        valfmt='%0.0f',
    )

    # Обновление графика
    def update(val):
        idx = int(slider.val)
        sc.set_offsets(frames[idx][0][:, :2])
        sc.set_array(frames[idx][1])
        ax.set_title(f"{title} (Кадр {idx})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    
if __name__ == "__main__":
    scenario_name, name = "square_drop", "2025.08.09.14.12.xyz"
    # В файле create_slider_animation.py
    xyz_path = Path(__file__).parent.parent / "results" / scenario_name / name

    create_slider_animation(
        data_source=str(xyz_path),  # Путь к файлу
        parameter="rho",  # Параметр для отображения
        cmap="viridis",  # Цветовая карта
        title="Капля жидкости (Плотность)",  # Заголовок
        figsize=(10, 6)  # Размер окна
    )
