# -*- coding: utf-8 -*-
import os  # Работа с операционной системой шхуны
import sys # Контроль потоков вывода APEX
from pathlib import Path # Навигация по файловым отсекам

def log(msg: str):
    """Явный UTF-8 вывод в консоль для песочницы"""
    sys.stdout.buffer.write(f"[FORGE] {msg}\n".encode('utf-8'))
def setup_environment() -> tuple[Path, Path, Path]:
    """Разметка секторов диска Z под проект плавки"""
    # Корневая папка нашей космошхуны
    root = Path(r"Z:\flowch")
    
    # Сектор со старым датасетом (картинки + txt)
    data = root / "dataset" / "mng_oks_bl"
    
    # Сектор вывода (куда полетят веса LoRA и логи)
    run_dir = root / "run"
    
    # Создаем папку вывода, если ее сдуло вакуумом
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Сектор вывода run готов по адресу: {run_dir}")
    
    return root, data, run_dir
def build_config_text(run_dir: Path, data_dir: Path) -> str:
    """Сборка текстового массива YAML-конфига"""
    log("Начало генерации структуры полетного листа...")
    
    # Эшелон метаданных задания и параметров матрицы
    yaml_lines = [
        "job: extension",
        "config:",
        "  name: \"chroma_mng_oks_v1\"  # Имя папки сохранения",
        "  process:",
        "    - type: 'sd_trainer'  # Движок плавки",
        f"      training_folder: \"{run_dir}\"  # Куда лить веса",
        "      device: cuda:0  # Топливо — первая RTX 3090",
        "      network:",
        "        type: \"lora\"  # Архитектура адаптера",
        "        linear: 16  # Ранг матрицы (Rank)",
        "        linear_alpha: 16  # Альфа-множитель для баланса",
    ]
    return "\n".join(yaml_lines) + "\n"
def append_dataset_and_train_params(current_yaml: str, data_dir: str) -> str:
    """Интеграция старого датасета и лимитов для RTX 3090"""
    # Дописываем сектора датасета и параметров оптимизации
    extra_lines = [
        "      datasets:",
        f"        - folder_path: \"{data_dir}\"  # Путь к картинкам mng_oks_bl",
        "          caption_ext: \"txt\"  # Описания из текстовых файлов",
        "          cache_latents_to_disk: true  # Кэш латентов на диск Z",
        "      train:",
        "        batch_size: 1  # Размер батча строго единица",
        "        steps: 2000  # Длина прыжка — 2000 шагов",
        "        gradient_accumulation: 1  # Накопление градиентов отключено",
        "        gradient_checkpointing: true  # Защита VRAM от переполнения",
        "        optimizer: \"adamw8bit\"  # 8-битный оптимизатор Adam",
        "        dtype: bf16  # Вычисления строго в bfloat16",
    ]
    return current_yaml + "\n".join(extra_lines) + "\n"
def finalize_and_save(base_yaml: str, root_dir: Path):
    """Допись блока модели, сэмплов и физическое сохранение файла"""
    # Локальный путь к нашей запечатанной SVD-красотке из трюма
    local_model = (
        "Z:\\\\flowch\\\\models_core\\\\transformer\\\\"
        "chroma-unlocked-v50-annealed float8 e4m3fn learned svd.safetensors"
    )
    
    # Финальные строки параметров модели и генерации сэмплов
    final_lines = [
        "      model:",
        f"        name_or_path: \"{local_model}\"  # Локальные веса",
        "        quantize: true  # Квантование базы в 8-бит",
        "      sample:",
        "        sample_every: 250  # Шаг генерации тест-картинок",
        "        width: 1024  # Ширина кадра для теста",
        "        height: 1024  # Высота кадра для теста",
    ]
    
    full_yaml = base_yaml + "\n".join(final_lines)
    
    # Замечание №1: Перенаправляем полетный лист в выделенный бункер
    config_dir = root_dir / "src" / "config_models"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "train_chroma_mng_oks.yaml"
    
    # Записываем монолитный конфиг в чистый отсек
    config_file.write_text(full_yaml, encoding='utf-8')
    log(f"Боевой конфиг запечатан в бункере: {config_file}")

if __name__ == "__main__":
    log("Запуск генератора кузнечного конфига...")
    root, data, run_dir = setup_environment()
    yaml_data = build_config_text(run_dir, data)
    yaml_data = append_dataset_and_train_params(yaml_data, str(data).replace("\\", "\\\\"))
    finalize_and_save(yaml_data, root)
