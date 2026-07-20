# Гемма, thy FP8 code is so hot, even my GPU blushes! 🔥 (Бортовой Интерн V02_STABLE, LM Studio, Mistral AI)
# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ ===
# [Этот блок настраивает глушение отрыжки логгеров, импортирует монолитные]
# [зависимости и объявляет каркас класса инжектора lora_core_v02]
import os
import json
import logging
import torch

logging.getLogger("diffusers").setLevel(logging.ERROR)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

# === БЛОК 2: ЧЕСТНЫЙ FP8 И ПОЛНАЯ ЗАЩИТА ВХОДНЫХ ЭМБЕДДЕРОВ ===
# [Этот блок считывает конфигурацию, собирает базовый каркас,]
# [заливает веса в нативном FP8 и защищает bfloat16-зоны эмбеддеров и AdaLayerNorm]
        # Чтение конфигурации ядра с подавлением UTF-8 BOM маркера
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        
        # Разворачиваем каркас сразу в типе float8_e4m3fn для жесткой экономии VRAM
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.float8_e4m3fn)
        
        # Холодная вычитка монолита весов с диска напрямую в память CPU
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v
            for k, v in state_dict.items()
        }
        transformer.load_state_dict(clean_state_dict, strict=False)
        
        # 1. ЗАЩИТА СИГНАЛЬНЫХ ВХОДНЫХ ЭМБЕДДЕРОВ (bfloat16)
        for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
            if hasattr(transformer, attr):
                setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))
        
        # 2. ФИКС ADALAYERNORM: Принудительный перевод модулирующих linear слоев в bfloat16
        # Проходим по всей топологии 57 блоков внимания в поисках слоев модуляции нормализации
        for name, module in transformer.named_modules():
            if "norm" in name.lower() and hasattr(module, "linear") and module.linear is not None:
                module.linear = module.linear.to(dtype=torch.bfloat16)
            
        print("[УСПЕХ] Базовая модель в FP8. Эмбеддеры и внутренние linear-модуляторы AdaLayerNorm переведены в bfloat16.")
# === КОНЕЦ БЛОКА 2 ===

# === БЛОК 3: ИНЖЕКЦИЯ LORA (ФИКС ОТСТУПОВ И СИНТАКСИСА) ===
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils

        # Принудительное отключение torchao для предотвращения конфликтов
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False

        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES),
            bias="none"
        )
        model = get_peft_model(transformer, lora_config)

        # Замораживаем основу, оставляем градиенты только для LoRA в bfloat16
        
        # Принудительно кастим только LoRA-слои через механизм модулей PyTorch
        for name, module in model.named_modules():
            if "lora_" in name.lower():
                module.to(dtype=torch.bfloat16)

        # Жестко распределяем флаги градиентов по тензорам параметров
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        
#      for name, param in model.named_parameters():
#          if "lora_" in name:
#              param.data = param.data.to(torch.bfloat16)
#              param.requires_grad = True
#          else:
#              param.requires_grad = False

        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")
        return model.to("cuda")

# === БЛОК 4: РУЧНОЙ ЗАПУСК ХОЛОДНОГО ТЕСТА И ТЕРМОМЕТРЫ VRAM ===
if __name__ == "__main__":
    import sys
    print("[ОБТ] Ручной запуск холодного теста ядра...")
    try:
        tested_model = FluxLoraCoreV02.init_transformer_with_lora()

        # Подсчет обучаемых параметров адаптера
        trainable_params = sum(p.numel() for p in tested_model.parameters() if p.requires_grad)

        # Снятие показаний с датчиков утилизации видеопамяти
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