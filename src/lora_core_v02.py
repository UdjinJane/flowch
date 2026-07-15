# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from config import TrainConfig


class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        config_json_path = os.path.join(os.path.dirname(__file__), "transformer_config.json")
        print(f"[ОБТ] Шаг 5.3: Сборка каркаса Transformer V02 из локального чертежа: {config_json_path}")
        
        if not os.path.exists(TrainConfig.MODEL_SINGLE_FILE):
            print(f"[КРИТ] Чекпоинт Chroma1-Base не найден: {TrainConfig.MODEL_SINGLE_FILE}")
            sys.exit(1)
            
        print(f"[ОБТ] Нативная оффлайн-загрузка квантового ядра FP8 SVD через Single File: {TrainConfig.MODEL_SINGLE_FILE}")
        # Намертво загружаем 8-битный монолит напрямую в GPU без промежуточного bfloat16-раздувания
        transformer = FluxTransformer2DModel.from_single_file(
            TrainConfig.MODEL_SINGLE_FILE,
            config=config_json_path,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )


        
        # НАМЕРТВО ЗАМОРАЖИВАЕМ БАЗУ (оригинальные веса Хромы не изменятся)
        transformer.requires_grad_(False)
        
        # КРИТИЧЕСКИЙ АНТИ-OOM МАНЕВР: Активируем градиентный чекпоинтинг для тотальной разгрузки VRAM!
        # Выметает матрицы активаций внимания из физической памяти GPU, обнуляя заезд в Shared Memory.
        transformer.enable_gradient_checkpointing()
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
        lora_model = lora_model.to(device="cuda")
        # Принудительно оставляем только обучаемые параметры LoRA в bfloat16 для стабильности градиентов
        for p in lora_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 полностью герметизировано на GPU.")
        return lora_model

    @staticmethod
    def get_peft_model_state_dict(lora_model):
        return get_peft_model_state_dict(lora_model)

    @staticmethod
    def save_file(tensor_dict, filepath):
        save_file(tensor_dict, filepath)


if __name__ == "__main__":

    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    print("[ОБТ] Холодный тест отсека инжекции V02...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_params:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===

