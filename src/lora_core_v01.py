import os
import sys
import json
import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from config import TrainConfig

class FluxLoraCoreV01:
    @staticmethod
    def init_transformer_with_lora():
        config_json_path = os.path.join(os.path.dirname(__file__), "transformer_config.json")
        print(f"[Т] Шаг 5.3: Сборка каркаса Transformer из чертежа с защитой от BOM: {config_json_path}")
        
        if not os.path.exists(TrainConfig.MODEL_SINGLE_FILE):
            print(f"[Т] екпоинт Chroma1-Base не найден: {TrainConfig.MODEL_SINGLE_FILE}")
            sys.exit(1)
            
        # 1. итаем локальный JSON конфигурации через utf-8-sig (выжигает BOM)
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
            
        # нициализируем пустую модель в мета-пространстве для экономии VRAM
        transformer = FluxTransformer2DModel.from_config(config_dict)
            
        # 2. ыдергиваем чистые веса из нашего одиночного safetensors
        print(f"[Т] агрузка физических весов трансформера из монолита: {TrainConfig.MODEL_SINGLE_FILE}")
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Срезаем префиксы, если чекпоинт упакован под ComfyUI
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                clean_state_dict[k.replace("model.diffusion_model.", "")] = v
            else:
                clean_state_dict[k] = v
                
        # Сажаем веса на каркас в bfloat16
        print("[Т] азвертывание матриц на тензорные ядра CUDA...")
        transformer = transformer.to_empty(device="cuda")
        
        # агружаем веса в режиме strict=False, так как в монолите Хромы лежат еще веса VAE/CLIP
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer = transformer.to(torch.bfloat16)
        
        # 3. Т   (веса Хромы не изменятся!)
        transformer.requires_grad_(False)
        
        # 4. ТЫ ТС PEFT/LORA
        print(f"[Т] нжекция LoRA адаптеров (Rank: {TrainConfig.LORA_RANK}, Alpha: {TrainConfig.LORA_ALPHA})...")
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=TrainConfig.TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian"
        )
        
        # рошиваем трансформер адаптерами
        lora_model = get_peft_model(transformer, lora_config)
        lora_model = lora_model.to(device="cuda", dtype=torch.bfloat16)
        
        print("[СХ] Ядро LoRA_Core_V01 успешно имплантировано и готово к плавке.")
        return lora_model

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    print("[Т] Тестирование отсека инжекции: lora_core_v01")
    lora_model = FluxLoraCoreV01.init_transformer_with_lora()
    
    # Считаем обучаемые параметры, чтобы убедиться, что база заморожена, а LoRA открыта
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print("--- [Т] ТСТ  LORA V01 СТЬ  ---")
    print(f"оличество активных (обучаемых) параметров LoRA: {trainable_params:,}")
