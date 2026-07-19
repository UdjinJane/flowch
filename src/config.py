import os

class TrainConfig:
    """
    Центральный распределительный щит конфигурации реактора Flux-LoRA.
    Все маршевые параметры и аппаратные лимиты шхуны зафиксированы здесь.
    """
    # --- Системные пути ---
    ROOT_DIR = "Z:\\flowch"
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    CACHE_DIR = os.path.join(ROOT_DIR, "cache")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    TEXT_EMBEDS_CACHE = os.path.join(CACHE_DIR, "text_embeds")
    LATENT_EMBEDS_CACHE = os.path.join(CACHE_DIR, "latent_embeds")
    
    # Стыковочные узлы для dataset_v02.py (Исправлено!)
    CACHE_TEXT_DIR = os.path.join(ROOT_DIR, "cache", "text_embeds")
    CACHE_LATENT_DIR = os.path.join(ROOT_DIR, "cache", "latent_embeds")

    # --- Параметры плавки (Гиперпараметры) ---
    NUM_EPOCHS = 1                        # Теперь контролируется отсюда! Никакого хардкода!
    LEARNING_RATE = 2e-5                  # Скорость обучения для стабильных весов AdamW
    GRADIENT_ACCUMULATION_STEPS = 4        # Накопление градиентов для виртуального увеличения батча
    SAVE_STEPS = 100                      # Рубеж сохранения чекпоинтов и запекания тест-кадров

    # --- Геометрия и физические ограничения ---
    RESOLUTION = 512                      # Фиксированное разрешение обработки (512px)
    VRAM_LIMIT_GB = 21.0                  # Жесткий потолок утилизации VRAM шхуны

    # --- Слой инжекции адаптеров ---
    TARGET_MODULES = [
        "to_q", "to_k", "to_v", "to_out.0"
    ]
