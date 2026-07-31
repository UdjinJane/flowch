# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

def log(msg: str):
    """Вывод логов сканирования в UTF-8"""
    sys.stdout.buffer.write(f"[SCANNER] {msg}\n".encode('utf-8'))

def scan_shuna():
    """Поиск весов моделей и аудит корневых отсеков диска Z"""
    root_drive = Path(r"Z:\flowch")
    log(f"Запуск эхолота. Прочесываем сектор: {root_drive}")
    
    # 1. Аудит папок верхнего уровня для понимания обстановки
    log("=== Структура директорий ")
    try:
        for item in root_drive.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                log(f" Обнаружен отсек: {item.name}/")
    except Exception as e:
        log(f"Ошибка доступа к корню: {str(e)}")
        
    # 2. Целевой поиск тяжелого вооружения (базовых моделей)
    log("=== Поиск файлов базовых моделей (.safetensors) ===")
    extensions = ['*.safetensors', '*.ckpt']
    
    # Ищем на глубину до 3 папок, чтобы не зависнуть в кэшах
    count = 0
    for ext in extensions:
        for path in root_drive.rglob(ext):
            # Игнорируем временные папки и кэш, если они есть
            if "checkpoint" in path.name.lower() or ".cache" in path.parts:
                continue
            size_gb = path.stat().st_size / (1024 ** 3)
            log(f" Найдена модель: {path.relative_to(root_drive)} | Размер: {size_gb:.2f} ГБ")
            count += 1
            if count >= 15: # Предохранитель, чтобы не спамить консоль
                log("Вывод ограничен предохранителем в 15 моделей.")
                break
                
    log("Сканирование сектора завершено.")

if __name__ == "__main__":
    scan_shuna()
