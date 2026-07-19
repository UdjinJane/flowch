import os

class TrainConfig:
    """Конфигурация реактора Flux-LoRA (Исправлено)"""
    ROOT_DIR = "Z:\\flowch"
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "mng_oks_bl")
    CACHE_DIR = os.path.join(ROOT_DIR, "cache")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
    
    # Исправленные пути для кэша
    CACHE_TEXT_DIR = os.path.join(CACHE_DIR, "text_embeds")
    CACHE_LATENT_DIR = os.path.join(CACHE_DIR, "latent_embeds")
    
    # Путь к физическому монолиту весов (добавлено)
    MODEL_SINGLE_FILE = os.path.join(ROOT_DIR, "base_model", "flux1-dev-fp8.safetensors")

    # Гиперпараметры и геометрия
    BATCH_SIZE = 1                        
    NUM_EPOCHS = 1                        
    LEARNING_RATE = 2e-5                  
    GRADIENT_ACCUMULATION_STEPS = 4        
    SAVE_STEPS = 100                      
    RESOLUTION = 512                      
    VRAM_LIMIT_GB = 21.0                  
    TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
