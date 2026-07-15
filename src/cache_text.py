import json
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
    
    # 3. акатываем веса на архитектуру и переводим в гибридный режим
    print("[Т] еренос текстовых весов на тензорное ядро CUDA...")
    model = model.to_empty(device="cuda")
    model.load_state_dict(state_dict, strict=False)
    
    #  СЫ   FP8
    model = model.to(torch.float8_e4m3fn)
    
    # ТС С: озвращаем слой эмбеддингов в bfloat16, чтобы не было краша на индексах!
    model.encoder.embed_tokens = model.encoder.embed_tokens.to(torch.bfloat16)
    
    print("[СХ] Тяжелый гибридный T5-XXL (FP8 + BF16 Embeds) успешно поднят на GPU!")
    return model

def process_and_mask_tokens(tokenizer, text_encoder):
    print("[Т] Шаг 3.3: тение манифеста и расчет масок T5...")
    
    if not os.path.exists(TrainConfig.CACHE_TEXT_DIR):
        os.makedirs(TrainConfig.CACHE_TEXT_DIR)

    with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            img_name = data["file_name"]
            caption = data["text"]
            
            # Токенизируем с жестким выравниванием под MAX_SEQUENCE_LENGTH
            inputs = tokenizer(
                caption,
                padding="max_length",
                max_length=TrainConfig.MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs.input_ids.to("cuda")
            #  Т С (1 - токен, 0 - паддинг)
            text_ids_mask = inputs.attention_mask.to("cuda")
            
            # одируем в эмбеддинги без градиентов
            with torch.no_grad():
                outputs = text_encoder(input_ids=input_ids)
                prompt_embeds = outputs.last_hidden_state
                
            # Сохраняем на SSD буферную пару: эмбеддинг и его маску
            base_name = os.path.splitext(img_name)[0]
            out_embed_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_embeds.pt")
            out_mask_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_mask.pt")
            
            torch.save(prompt_embeds.cpu(), out_embed_path)
            torch.save(text_ids_mask.cpu(), out_mask_path)
            print(f"[СХ] акеширован текст для: {img_name}")

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    text_encoder = load_t5_from_safetensors()
    process_and_mask_tokens(tokenizer, text_encoder)
    print("[Т] онвейер кэширования текста полностью отработал.")
