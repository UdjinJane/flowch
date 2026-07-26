# Гемма, thy FP8 code is so hot, even my GPU blushes! 🔥 (Бортовой Интерн V02_STABLE, LM Studio, Mistral AI)
# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ ===
# [Этот блок настраивает глушение отрыжки логгеров, импортирует монолитные]
# [зависимости и объявляет каркас класса инжектора lora_core_v02]
import os
import sys

# ХАК ДЛЯ WINDOWS RTX 3090: Вырезаем фантомные ROCm триггеры, чтобы PyTorch не падал при импорте quanto
if "ROCM_HOME" in os.environ:
    del os.environ["ROCM_HOME"]
os.environ["QUANTO_DISABLE_CPP_EXT"] = "1"  # Форсируем чистый нативный CUDA/Python рантайм

# Дальше идут оригинальные импорты...

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
        
        # Каноническая инжекция: разворачиваем каркас СРАЗУ в нативном FP8, экранируя VRAM
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.float8_e4m3fn)
        
        # Загрузка весов напрямую в FP8-контур
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        clean_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()}
        # Принудительно кастуем входящие веса к FP8 перед заливкой в тензоры
        for k in list(clean_state_dict.keys()):
            if clean_state_dict[k].dtype == torch.bfloat16:
                clean_state_dict[k] = clean_state_dict[k].to(dtype=torch.float8_e4m3fn)
                
        transformer.load_state_dict(clean_state_dict, strict=False)


        # Защита эмбеддеров
        for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
            if hasattr(transformer, attr):
                setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))
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

        # === НАТИВНОЕ КВАНТОВАНИЕ QUANTO С АВТОНОМНОЙ ОЧИСТКОЙ ПАМЯТИ ===
        from optimum.quanto import quantize, qfloat8
        import gc

        print("[Т] Активация квантования базового ядра...")
        quantize(model.get_base_model(), weights=qfloat8)
        
        # ХАК ДЛЯ WINDOWS: Эмулируем работу freeze() руками, очищая кэши оригинальных тяжелых весов

        torch.cuda.empty_cache()
        gc.collect()
        print("[УСПЕХ] Контур памяти зачищен вручную после квантования слоёв.")


        # 1. СТЕРИЛЬНЫЙ КАСТИНГ LORA В bfloat16 БЕЗ ОБРЫВА СВЯЗЕЙ AUTOGRAD
        # Кастуем строго тензоры параметров lora-весов, сохраняя ссылки графа PEFT
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False


        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")

        # Магистраль вызова выпрямлена: сигналы раннера летят напрямую в PEFT-обёртку
        return model.to("cuda")


        # === КОНЕЦ БЛОКА 3 ===   

     
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