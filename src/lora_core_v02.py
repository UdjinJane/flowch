# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict as peft_get_state_dict
# ... (новые импорты)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig
# Инжектируем нативный квантовый инструмент PyTorch Foundation
from torchao.quantization import quantize_, float8_weight_only



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
        
        print("[ОБТ] Сборка каркаса Transformer V02...")
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

        print("[ОБТ] Запуск нативного квантования TorchAO FP8 (Изоляция x_embedder)...")
        # Хирургический фильтр: квантуем все Linear слои, КРОМЕ входного эмбеддера x_embedder
        def filter_fn(mod, name):
            if "x_embedder" in name:
                return False
            return isinstance(mod, torch.nn.Linear)

        # Переводим веса базовой Хромы в float8 на месте прямо внутри VRAM
        # === РАБОЧИЙ БЛОК: ИНЖЕКЦИЯ LORA ДО КВАНТОВАНИЯ СТАРТ ===
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer = transformer.to(torch.bfloat16)

        # 1. Сначала LoRA на bf16
        lora_model = get_peft_model(transformer, lora_config)

        # 2. Квантование (игнорируя 'lora_' в фильтре)
        def filter_fn(mod, name):
            if "x_embedder" in name or "lora_" in name: return False
            return isinstance(mod, torch.nn.Linear)
        quantize_(lora_model, float8_weight_only(), filter_fn)

        # 3. Фиксация
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()
        lora_model = lora_model.to(device="cuda")
# === РАБОЧИЙ БЛОК: ИНЖЕКЦИЯ LORA ДО КВАНТОВАНИЯ ФИНАЛ ===

        lora_model = lora_model.to(device="cuda")
        # Принудительно оставляем только обучаемые параметры LoRA в bfloat16 для стабильности градиентов
        for p in lora_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 полностью герметизировано на GPU.")
        return lora_model

    @staticmethod
    def get_peft_model_state_dict(lora_model):
        return peft_get_state_dict(lora_model)

    @staticmethod
    def save_file(tensor_dict, filepath):
        st_save_file(tensor_dict, filepath)



if __name__ == "__main__":

    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    print("[ОБТ] Холодный тест отсека инжекции V02...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_params:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===

