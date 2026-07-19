# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ ===
# [Этот блок настраивает глушение отрыжки логгеров, импортирует монолитные]
# [зависимости и объявляет каркас класса инжектора lora_core_v02]
import os
import json
import logging
import torch

# Намертво блокируем вывод предупреждений diffusers о геометрии и sample_size
logging.getLogger("diffusers").setLevel(logging.ERROR)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")
        
        # Сброс и прогрев встроенного термометра видеопамяти
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
# === КОНЕЦ БЛОКА 1 ===

# === БЛОК 2: ЧЕСТНЫЙ FP8 И ЗАЩИТА СИГНАЛЬНЫХ ЗОН ЭМБЕДДЕРОВ ===
# [Этот блок считывает конфигурацию, собирает базовый каркас,]
# [заливает веса в нативном FP8 и защищает bfloat16-зоны x_embedder и time_text_embed]
        # Безопасное чтение конфигурации с подавлением UTF-8 BOM маркера
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        
        # Razворачиваем каркас сразу в типе float8_e4m3fn для жесткой экономии VRAM
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.float8_e4m3fn)
        
        # Холодная вычитка монолита весов с диска напрямую в память CPU
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очистка префиксов ключей ComfyUI с сохранением нативного FP8 типа данных
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v
            for k, v in state_dict.items()
        }
        
        # Накатываем веса на FP8 каркас
        transformer.load_state_dict(clean_state_dict, strict=False)
        
        # ХАРДКОРНЫЙ ФИКС: Принудительно возвращаем критические эмбеддеры в bfloat16 (требование Flux)
        if hasattr(transformer, "x_embedder"):
            transformer.x_embedder = transformer.x_embedder.to(dtype=torch.bfloat16)
            
        if hasattr(transformer, "time_text_embed"):
            transformer.time_text_embed = transformer.time_text_embed.to(dtype=torch.bfloat16)
            
        print("[УСПЕХ] Базовая модель герметизирована в честном FP8. Сигнальные зоны x_embedder и time_text_embed переведены в bfloat16.")
# === КОНЕЦ БЛОКА 2 ===


# === БЛОК 3: ИНЖЕКЦИЯ LORA С ПРИНУДИТЕЛЬНЫМ ВЫКЛЮЧЕНИЕМ TORCHAO ===
# [Этот блок настраивает PEFT-адаптеры, блокирует конфликты с torchao]
# [и активирует bfloat16-точность исключительно для обучаемых LoRA весов]
        # Принудительная блокировка внутренних авто-квантов PEFT, которые ломают граф
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False
        
        # Конфигурация посадочных мест LoRA адаптера под мишени из TrainConfig
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES),
            bias="none"
        )
        model = get_peft_model(transformer, lora_config)
        
        # Точечный перевод LoRA хвостов в bfloat16 и заморозка основного FP8 каркаса
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")
        return model.to("cuda")
# === КОНЕЦ БЛОКА 3 ===

# === БЛОК 4: РУЧНОЙ ЗАПУСК ХОЛОДНОГО ТЕСТА И ТЕРМОМЕТРЫ VRAM ===
# [Этот блок содержит точку входа для автономного тестирования ядра,]
# [запускает инжектор и опрашивает встроенные датчики выделенной памяти]
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
# === КОНЕЦ БЛОКА 4 ===
