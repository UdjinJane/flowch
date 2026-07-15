import os

class TrainConfig:
    # --- ССТЫ Т ---
    ROOT_DIR = r"Z:\flowch"
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    
    # уферная зона для кэша тензоров
    CACHE_TEXT_DIR = os.path.join(ROOT_DIR, "cache", "text_embeds")
    CACHE_LATENT_DIR = os.path.join(ROOT_DIR, "cache", "latent_embeds")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    # --- С   CHROMA1 ---
    # апитан, укажите здесь точный локальный путь к весам Chroma1 / FLUX в вашей системе!
    MODEL_PATH = r"Z:\models\Chroma1-Base" 
    
    # --- ТЫ ТТ T5-XXL ---
    MAX_SEQUENCE_LENGTH = 256  # птимальное окно для Chroma/FLUX
    
    # --- ТЫ Т Я ---
    BATCH_SIZE = 1             # естко зафиксировано по перфокарте для контроля температуры ядра
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = "constant"
    MAX_TRAIN_STEPS = 1500
    
    # --- ТЫ LORA (PEFT) ---
    LORA_RANK = 16
    LORA_ALPHA = 16
    TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"] # сновные узлы инжекции в 48 слоев

print("[Т] онфигурация ядра ядра зафиксирована.")
