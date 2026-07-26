# =========================================================================
# КРИСТАЛЬНО СТЕРИЛЬНОЕ bfloat16-ЯДРО ИНЖЕКТОРА LoRA (ВЕРСИЯ V02_STABLE)
# =========================================================================
# [Этот блок импортирует чистый PyTorch и Diffusers, полностью исключая quanto]

import os
import sys
import json
import logging
import torch

# Полное глушение внутренних предупреждений Diffusers для чистоты логов
logging.getLogger("diffusers").setLevel(logging.ERROR)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02 (Чистый bfloat16)")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Чтение конфигурации ядра с гарантированным подавлением UTF-8 BOM маркера
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)

        # Каноническая сборка: разворачиваем каркас СРАЗУ в аппаратном bfloat16
        print("[Т] Сборка каркаса модели в нативном bfloat16...")
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.bfloat16)

        # Безопасная вычитка весов с SSD
        print("[Т] Загрузка safetensors-монолита в RAM...")
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        clean_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()}
        
        # Принудительное выпрямление входящих весов к bf16 перед посадкой в слои
        for k in list(clean_state_dict.keys()):
            if clean_state_dict[k].dtype != torch.bfloat16:
                clean_state_dict[k] = clean_state_dict[k].to(dtype=torch.bfloat16)

        print("[Т] Интеграция весов в ядро трансформера...")
        transformer.load_state_dict(clean_state_dict, strict=False)

        # Аппаратная изоляция шлюзов эмбеддеров
        for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
            if hasattr(transformer, attr):
                setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))

        # Блокировка torchao и инициализация LoRA
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False

        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES),
            bias="none"
        )
        model = get_peft_model(transformer, lora_config)

        # Обучение только LoRA весов в bfloat16
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model.to("cuda")

# === БЛОК 4: ХОЛОДНЫЙ ТЕСТ И МОНИТОРИНГ VRAM ===
if __name__ == "__main__":
    try:
        model = FluxLoraCoreV02.init_transformer_with_lora()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Мониторинг VRAM
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[ОТК] УСПЕХ | VRAM: {allocated:.2f} GB | LoRA params: {trainable_params:,}")
        else:
            print("[ОТК] Тест пройден, CUDA неактивна")
    except Exception as e:
        print(f"[АВАРИЯ] {str(e)}", file=sys.stderr)
        sys.exit(1)
