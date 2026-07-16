import os

class TrainConfig:
    # --- МОДУЛЬ АВТОНОМНОЙ НАВИГАЦИИ (ОТНОСИТЕЛЬНЫЕ ПУТИ) ---
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # --- МАГИСТРАЛИ ДАННЫХ И КЭША ---
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    CACHE_TEXT_DIR = os.path.join(ROOT_DIR, "cache", "text_embeds")
    CACHE_LATENT_DIR = os.path.join(ROOT_DIR, "cache", "latent_embeds")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

    # --- СУНДУЧОК CORE-МОДЕЛЕЙ (ВНУТРИ ПЕСОЧНИЦЫ) ---
    CORE_MODELS_DIR = os.path.join(ROOT_DIR, "models_core")
    
    # Нацеливаем на основной "полтинник" Chroma
    MODEL_SINGLE_FILE = os.path.join(CORE_MODELS_DIR, "transformer", "chroma-unlocked-v50-annealed_float8_e4m3fn_learned_svd.safetensors")
    T5_ENCODER_PATH = os.path.join(CORE_MODELS_DIR, "text_encoder", "t5xxl_bf16.safetensors")
    VAE_PATH = os.path.join(CORE_MODELS_DIR, "vae", "flux-vae-bf16.safetensors")

    # --- ПАРАМЕТРЫ ---
    MAX_SEQUENCE_LENGTH = 256
    RESOLUTION = 512
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    MAX_TRAIN_STEPS = 1500
    LORA_RANK = 16
    LORA_ALPHA = 16
    TARGET_MODULES = ["to_q.0", "to_k.0", "to_v.0", "to_out.0"]


