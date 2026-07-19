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

# === БЛОК 2: ЧЕСТНЫЙ FP8 И ПОЛНАЯ ЗАЩИТА ВХОДНЫХ ЭМБЕДДЕРОВ ===
# Чтение конфигурации и инициализация модели в FP8
config_dict = json.load(open(os.path.join(TrainConfig.SRC_DIR, "transformer_config.json"), "r", encoding="utf-8-sig"))
transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.float8_e4m3fn)

# Загрузка и очистка весов
state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
clean_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()}
transformer.load_state_dict(clean_state_dict, strict=False)

# Фикс эмбеддеров (перевод в bfloat16 для стабильности)
for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
    if hasattr(transformer, attr):
        setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))

print("[УСПЕХ] Модель в FP8, входные эмбеддеры переведены в bfloat16.")
# === КОНЕЦ БЛОКА 2 ===

# === БЛОК 3: ИНЖЕКЦИЯ LORA С ПРИНУДИТЕЛЬНЫМ ВЫКЛЮЧЕНИЕМ TORCHAO ===
# [Отключает конфликты с torchao, настраивает PEFT/LoRA и включает bfloat16]
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
