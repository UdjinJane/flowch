# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import json
import logging
import torch

# Гасим отрыжку логгера diffusers до инициализации модели
logging.getLogger("diffusers").setLevel(logging.ERROR)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")
        
        # Включаем встроенный термометр VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # === БЛОК 2: БЕЗОПАСНОЕ ЧТЕНИЕ И КАСТ В BF16 ===
        # Считывание конфигурации ядра с подавлением UTF-8 BOM маркера
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        
        # Сборка чистого каркаса в ОЗУ
        transformer = FluxTransformer2DModel.from_config(config_dict)
        
        # Вычитка монолита весов с диска напрямую в CPU
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очистка префиксов ключей ComfyUI и принудительный каст тензоров в bfloat16
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v.to(torch.bfloat16)
            for k, v in state_dict.items()
        }
        
        # Заливка очищенных bfloat16 весов в созданный каркас
        transformer.load_state_dict(clean_state_dict, strict=False)
        print("[УСПЕХ] Базовая модель собрана в ОЗУ без раздутия весов.")

        # === БЛОК 3: ИНЖЕКЦИЯ LORA И ФИНАЛИЗАЦИЯ ===
        # Блокировка конфликтов внутри PEFT с библиотекой torchao
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False
        
        # Конфигурация целевых мишеней адаптера
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK, 
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES), 
            bias="none"
        )
        model = get_peft_model(transformer, lora_config)
        
        # Заморозка базовой сети и точечная активация градиентов LoRA
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")
        return model.to("cuda")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===
if __name__ == "__main__":
    import sys
    print("[ОБТ] Ручной запуск холодного теста ядра...")
    try:
        tested_model = FluxLoraCoreV02.init_transformer_with_lora()
        
        # Подсчет параметров и опрос термометров VRAM
        trainable_params = sum(p.numel() for p in tested_model.parameters() if p.requires_grad)
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_report = f"| VRAM Текущая: {allocated:.2f} GB | Пик: {peak:.2f} GB"
        else:
            mem_report = "| CUDA недоступна"

        print(f"[ОТК] ТЕСТ ПРОЙДЕН УСПЕШНО! {mem_report}")
        print(f"[ОТК] Активных LoRA мишеней в bf16: {trainable_params:,}")
        
    except Exception as e:
        print(f"[АВАРИЯ] Ядро выбросило критическое исключение: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

