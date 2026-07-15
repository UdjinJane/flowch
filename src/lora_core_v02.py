# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        config_json_path = os.path.join(os.path.dirname(__file__), "transformer_config.json")
        print(f"[ОБТ] Шаг 5.3: Сборка каркаса Transformer V02 из локального чертежа: {config_json_path}")
        
        if not os.path.exists(TrainConfig.MODEL_SINGLE_FILE):
            print(f"[КРИТ] Чекпоинт Chroma1-Base не найден: {TrainConfig.MODEL_SINGLE_FILE}")
            sys.exit(1)
            
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
            
        transformer = FluxTransformer2DModel.from_config(config_dict)
            
        print(f"[ОБТ] Загрузка физических весов трансформера из монолита: {TrainConfig.MODEL_SINGLE_FILE}")
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                clean_state_dict[k.replace("model.diffusion_model.", "")] = v
            else:
                clean_state_dict[k] = v
                
        print("[ОБТ] Развертывание базовых матриц Хромы на тензорные ядра CUDA...")
        transformer = transformer.to_empty(device="cuda")
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer = transformer.to(torch.bfloat16)
        
        # НАМЕРТВО ЗАМОРАЖИВАЕМ БАЗУ (оригинальные веса Хромы не изменятся)
        transformer.requires_grad_(False)
        
        # КРИТИЧЕСКИЙ АНТИ-OOM МАНЕВР: Активируем градиентный чекпоинтинг для тотальной разгрузки VRAM!
        # Выметает матрицы активаций внимания из физической памяти GPU, обнуляя заезд в Shared Memory.
        transformer.gradient_checkpointing_enable()
        print("[ОТК] Аппаратный градиентный чекпоинтинг успешно активирован в слоях трансформера.")
        
        print(f"[ОБТ] Инжекция LoRA адаптеров V02 (Rank: {TrainConfig.LORA_RANK}, Alpha: {TrainConfig.LORA_ALPHA})...")
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=TrainConfig.TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian"
        )
        
        lora_model = get_peft_model(transformer, lora_config)
        lora_model = lora_model.to(device="cuda", dtype=torch.bfloat16)
        
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 полностью герметизировано на GPU.")
        return lora_model

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    print("[ОБТ] Холодный тест отсека инжекции V02...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_params:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===

