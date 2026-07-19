import os

class TrainConfig:
    """Конфигурация реактора Flux-LoRA (Исправлено)"""
    ROOT_DIR = "Z:\\flowch"
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    CACHE_DIR = os.path.join(ROOT_DIR, "cache")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    MODELS_CORE_DIR = os.path.join(ROOT_DIR, "models_core") # Объявлено строго здесь!
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    
    # Исправленные пути для кэша
    CACHE_TEXT_DIR = os.path.join(CACHE_DIR, "text_embeds")
    CACHE_LATENT_DIR = os.path.join(CACHE_DIR, "latent_embeds")
    
    # --- Путь к отсеку логов со скриншота ---
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # --- Пути к локальным моделям (согласно скриншотам) ---
    # 1. Трансформер Chroma1
    MODEL_SINGLE_FILE = os.path.join(
        MODELS_CORE_DIR, "transformer", "chroma-unlocked-v50-annealed_float8_e4m3fn_learned_svd.safetensors"
    )
    # 2. Текстовый энкодер
    TEXT_ENCODER_PATH = os.path.join(
        MODELS_CORE_DIR, "text_encoder", "t5xxl_bf16.safetensors"
    )
    # 3. VAE
    VAE_PATH = os.path.join(
        MODELS_CORE_DIR, "vae", "flux-vae-bf16.safetensors"
    )

    # Гиперпараметры и геометрия
    BATCH_SIZE = 1                        
    NUM_EPOCHS = 1                        
    LEARNING_RATE = 2e-5                  
    GRADIENT_ACCUMULATION_STEPS = 4        
    SAVE_STEPS = 100                      
    RESOLUTION = 512                      
    VRAM_LIMIT_GB = 21.0                  
    TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
