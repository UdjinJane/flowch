import os

# Запрещаем фрагментацию, выставляем агрессивный сбор мусора и отключаем превентивный оффлоад WDDM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:256,"
    "garbage_collection_threshold:0.6,"
    "expandable_segments:True"
)


class TrainConfig:
    """
    Центральный конфигурационный щит Flux-LoRA.
    Синхронизирован с логикой ядра v02 и автоматической навигацией Кэпа.
    """
    # --- МОДУЛЬ АВТОНОМНОЙ НАВИГАЦИИ (ОТНОСИТЕЛЬНЫЕ ПУТИ) ---
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SRC_DIR)
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    # --- СТЕРИЛЬНЫЙ ОТСЕК ЛОГОВ (СОГЛАСНО СНИМКУ ЭКРАНА) ---
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    # --- МАГИСТРАЛИ ДАННЫХ И КЭША (ДЛЯ DATASET_V02) ---
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    CACHE_DIR = os.path.join(ROOT_DIR, "cache")
    CACHE_TEXT_DIR = os.path.join(CACHE_DIR, "text_embeds")
    CACHE_LATENT_DIR = os.path.join(CACHE_DIR, "latent_embeds")

    # --- СУНДУЧОК CORE-МОДЕЛЕЙ (МАРШРУТЫ К ВЕСАМ CHROMA1) ---
    MODELS_CORE_DIR = os.path.join(ROOT_DIR, "models_core")
    
    MODEL_SINGLE_FILE = os.path.join(
        MODELS_CORE_DIR, "transformer", "chroma-unlocked-v50-annealed_float8_e4m3fn_learned_svd.safetensors"
    )
    TEXT_ENCODER_PATH = os.path.join(
        MODELS_CORE_DIR, "text_encoder", "t5xxl_bf16.safetensors"
    )
    VAE_PATH = os.path.join(
        MODELS_CORE_DIR, "vae", "flux-vae-bf16.safetensors"
    )

    # --- ПАРАМЕТРЫ ПЛАВКИ (ГИПЕРПАРАМЕТРЫ) ---
    MAX_SEQUENCE_LENGTH = 256
    RESOLUTION = 512
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 2  # Удержание стабильности лосса из репозитория
    LEARNING_RATE = 2e-5
    MAX_TRAIN_STEPS = 1500
    SAVE_STEPS = 100
    NUM_EPOCHS = 40
    LORA_RANK = 16
    LORA_ALPHA = 16
    
    # Снайперские мишени LoRA под diffusers логику lora_core_v02.py
    TARGET_MODULES = ["to_q.0", "to_k.0", "to_v.0", "to_out.0"]
    
    # Физические ограничения шхуны
    VRAM_LIMIT_GB = 21.0

    # === МОДУЛЬ ВИРТУАЛЬНОГО ПОЛИГОНА APEX ===
    # True - активировать эмуляторы "черного ящика" для холостой отладки осей 
    # False - боевой режим плавки с реальной загрузкой весов VAE и трансформера
    USE_EMULATORS: bool = False  


