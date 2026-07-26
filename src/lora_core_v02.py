import os
import sys
import json
import logging
import gc
import torch
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

# ============================================================================
# СНАЙПЕРСКИЙ ХАК ДЛЯ WINDOWS: Динамическое ослепление PEFT без слепых импортов
# ============================================================================
import peft.utils

# 1. Глушим коренную функцию в утилитах
peft.utils.is_torchao_available = lambda *args, **kwargs: True

# 2. Экранируем системный кэш модулей, чтобы предотвратить ModuleNotFoundError
if "peft.utils" in sys.modules:
    sys.modules["peft.utils"].is_torchao_available = lambda *args, **kwargs: True
    
# Обманываем внутренний импорт PEFT, подставляя корень вместо отсутствующего import_utils
sys.modules["peft.utils.import_utils"] = sys.modules["peft.utils"]
# ============================================================================





class FluxLoraCoreV02:

    @staticmethod

    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02 (Нативный TorchAO)")
        
        # Сброс пиковых счетчиков CUDA для прецизионного контроля памяти
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Используем torchao для int8 квантования
        from torchao.quantization import quantize_, int8_weight_only

        # Загрузка конфига и модели (bfloat16)
        with open(os.path.join(TrainConfig.SRC_DIR, "transformer_config.json"), "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.bfloat16)

        # Вычитка весов и вставка
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        clean_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()}
        transformer.load_state_dict(clean_state_dict, strict=False)

        # Квантование (int8_weight_only)
        quantize_(transformer, int8_weight_only())

        # Чистка VRAM
        torch.cuda.empty_cache()
        gc.collect()

        # Изоляция эмбеддеров в bf16
        for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
            if hasattr(transformer, attr):
                setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))
        # Конфигурируем PEFT LoRA и внедряем в квантованное ядро [1.10]
        lora_config = LoraConfig(r=TrainConfig.LORA_RANK, lora_alpha=TrainConfig.LORA_ALPHA, target_modules=list(TrainConfig.TARGET_MODULES), bias="none")
        model = get_peft_model(transformer, lora_config)

        # Активируем градиенты только для LoRA-модулей в bfloat16 [1.10]
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model.to("cuda")

# === БЛОК 4: ХОЛОДНЫЙ ТЕСТ И МОНИТОРИНГ VRAM ===
if __name__ == "__main__":
    # Инициализация, проверка параметров и замер потребления VRAM [1.10]
    tested_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in tested_model.parameters() if p.requires_grad)
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Активных LoRA мишеней: {trainable_params:,} | VRAM: {allocated:.2f} GB")
