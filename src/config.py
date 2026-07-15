import os

class TrainConfig:
    # --- ССТЫ Т ---
    ROOT_DIR = r"Z:\flowch"
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    
    CACHE_TEXT_DIR = os.path.join(ROOT_DIR, "cache", "text_embeds")
    CACHE_LATENT_DIR = os.path.join(ROOT_DIR, "cache", "latent_embeds")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    # --- СТТ Т  COMFYUI ---
    MODEL_SINGLE_FILE = r"Z:\AiModels\models\diffusion_models\Chroma1-Base.safetensors"
    
    # аш целевой энкодер в папке клипов
    T5_ENCODER_PATH = r"Z:\AiModels\models\clip\t5xxl_fp8_e4m3fn.safetensors"
    
    # --- ТЫ ТТ T5-XXL ---
    TOKENIZER_DIR = r"Z:\flowch\src\tokenizer"
    MAX_SEQUENCE_LENGTH = 256  # кно токенов под Chroma1/FLUX
    
    # --- ТЫ Т Я ---
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = "constant"
    MAX_TRAIN_STEPS = 1500
    
    LORA_RANK = 16
    LORA_ALPHA = 16
    TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

print("[Т] онфигурация путей ComfyUI успешно обновлена.")
