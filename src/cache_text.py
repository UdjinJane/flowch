import os
import sys
import torch
from transformers import T5TokenizerFast, T5EncoderModel, T5Config
from config import TrainConfig

def get_tokenizer():
    # сли локального токенизатора нет в проекте - выкачиваем его правила
    if not os.path.exists(TrainConfig.TOKENIZER_DIR) or not os.listdir(TrainConfig.TOKENIZER_DIR):
        print("[Т] окальный токенизатор не найден. иксируем из сети в проект...")
        tokenizer = T5TokenizerFast.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="tokenizer"
        )
        tokenizer.save_pretrained(TrainConfig.TOKENIZER_DIR)
        print(f"[СХ] Токенизатор намертво вшит в папку: {TrainConfig.TOKENIZER_DIR}")
    else:
        print(f"[Т] агрузка токенизатора из локального отсека проекта...")
        tokenizer = T5TokenizerFast.from_pretrained(TrainConfig.TOKENIZER_DIR)
    return tokenizer
from safetensors.torch import load_file

def load_t5_from_safetensors():
    print(f"[Т] Шаг 3.2: агрузка тяжелого T5-XXL из закромов: {TrainConfig.T5_ENCODER_PATH}")
    
    # 1. Строим пустую дефолтную архитектуру T5-XXL v1.1
    config = T5Config.from_pretrained("google/t5-v1_1-xxl")
    
    # ежим низкого потребления VRAM: инициализируем пустую модель (без весов в RAM)
    with torch.device("meta"):
        model = T5EncoderModel(config)
        
    # 2. изически выдергиваем веса из нашего одиночного safetensors
    state_dict = load_file(TrainConfig.T5_ENCODER_PATH, device="cpu")
    
    # 3. акатываем веса на архитектуру и переводим в fp8 на CUDA
    print("[Т] еренос текстовых весов на тензорное ядро CUDA...")
    model = model.to_empty(device="cuda")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.float8_e4m3fn) 
    
    print("[СХ] Тяжелый T5-XXL fp8 успешно поднят на GPU!")
    return model

# страиваем вызов в конец исполняемого блока для теста
if __name__ == "__main__":
    tokenizer = get_tokenizer()
    text_encoder = load_t5_from_safetensors()
    print("[СХ] лок 2 прошел боевое крещение.")
