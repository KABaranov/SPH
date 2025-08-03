import os
from datetime import datetime
from typing import TextIO


def create_xyz_file(test_case: str) -> TextIO:
    """
    Создаёт файл для сохранения результатов в формате гггг.мм.дд.чч.мм.xyz

    Алгоритм:
    1. Если нет папки 'results', создаёт её
    2. В папке 'results' создаёт подпапку с именем тест-кейса (если её нет)
    3. В этой папке создаёт файл с текущей датой-временем в формате гггг.мм.дд.чч.мм.xyz
    4. Возвращает файловый объект, открытый в режиме добавления ('a')

    Args:
        test_case: Название тест-кейса (будет использовано как имя подпапки)

    Returns:
        Файловый объект, открытый в режиме добавления
    """
    # Создаём необходимые директории
    os.makedirs(f"results/{test_case}", exist_ok=True)

    # Формируем имя файла с текущей датой-временем
    current_time = datetime.now().strftime("%Y.%m.%d.%H.%M")
    filename = f"results/{test_case}/{current_time}.xyz"

    # Открываем файл в режиме добавления (encoding='utf-8' для корректной работы с Unicode)
    file = open(filename, mode='a', encoding='utf-8')

    return file
