import os
import sys
import json
import torch
from transformers import T5TokenizerFast, T5EncoderModel, T5Config
from safetensors.torch import load_file
from config import TrainConfig

def get_tokenizer():
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

def load_t5_from_safetensors():
    print(f"[Т] Шаг 3.2: агрузка тяжелого T5-XXL из закромов: {TrainConfig.T5_ENCODER_PATH}")
    
    # 1. Строим архитектурный каркас T5-XXL v1.1
    config = T5Config.from_pretrained("google/t5-v1_1-xxl")
    
    # нициализируем пустую модель в мета-пространстве для экономии RAM
    with torch.device("meta"):
        model = T5EncoderModel(config)
        
    # 2. звлекаем чистые 16-битные веса из нашего нового safetensors
    state_dict = load_file(TrainConfig.T5_ENCODER_PATH, device="cpu")
    
    # 3. Сажаем веса на каркас и принудительно разворачиваем СТЬЫ BFLOAT16 контур на GPU
    print("[Т] еренос текстовых весов в стабильный bfloat16 контур CUDA...")
    model = model.to_empty(device="cuda")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    
    print("[СХ] Тяжелый T5-XXL в режиме чистого bfloat16 успешно поднят на GPU!")
    return model

def process_and_mask_tokens(tokenizer, text_encoder):
    print("[Т] Шаг 3.3: тение манифеста и расчет масок T5...")
    
    if not os.path.exists(TrainConfig.CACHE_TEXT_DIR):
        os.makedirs(TrainConfig.CACHE_TEXT_DIR)

    with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            img_name = data["file_name"]
            caption = data["text"]
            
            # Токенизируем с жестким выравниванием под MAX_SEQUENCE_LENGTH (256)
            inputs = tokenizer(
                caption,
                padding="max_length",
                max_length=TrainConfig.MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs.input_ids.to("cuda")
            text_ids_mask = inputs.attention_mask.to("cuda")
            
            # одируем без расчета градиентов
            with torch.no_grad():
                outputs = text_encoder(input_ids=input_ids)
                prompt_embeds = outputs.last_hidden_state
                
            # звлекаем чистое имя кадра для сохранения тензоров
            base_name = os.path.splitext(img_name)[0]
            out_embed_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_embeds.pt")
            out_mask_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_mask.pt")
            
            # Сохраняем готовые монолиты на SSD буфер
            torch.save(prompt_embeds.cpu(), out_embed_path)
            torch.save(text_ids_mask.cpu(), out_mask_path)
            print(f"[СХ] акеширован текст и маска для: {img_name}")

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    text_encoder = load_t5_from_safetensors()
    process_and_mask_tokens(tokenizer, text_encoder)
    print("[СХ] онвейер кэширования текста СТЬ   отработал.")
