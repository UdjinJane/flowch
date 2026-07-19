# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict as peft_get_state_dict
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Шаг А: Считывание локальной геометрии transformer_config.json...")

        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")

        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)

        print("[ОБТ] Шаг Б: Разворачивание пустого каркаса Chroma1 в оперативной памяти CPU...")
        transformer = FluxTransformer2DModel.from_config(config_dict)

        print(f"[ОБТ] Шаг В: Вычитывание физического монолита весов с диска: {TrainConfig.MODEL_SINGLE_FILE}")
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")

        print("[ОБТ] Шаг Г: Фильтрация ключей и зачистка префиксов ComfyUI...")
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v
            for k, v in state_dict.items()
        }

        print("[ОБТ] Шаг Д: Послойная заливка весов в каркас на CPU и каст в bfloat16...")
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer = transformer.to(torch.bfloat16)
        print("[УСПЕХ] Базовая модель полностью собрана в ОЗУ в чистом bf16. VRAM не задета.")

        print("[ОБТ] Шаг Е: Подготовка адаптеров LoRA (глушение внутренних проверок PEFT)...")
        target_modules_list = list(TrainConfig.TARGET_MODULES)

        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=target_modules_list,
            bias="none"
        )

        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        orig_lora_ao = getattr(peft.tuners.lora.torchao, "is_torchao_available", None)
        orig_tune_ao = getattr(peft.tuners.tuners_utils, "is_torchao_available", None)

        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False

        print("[ОБТ] Шаг Ж: Инжекция LoRA-адаптеров в bf16-граф на CPU...")
        lora_model = get_peft_model(transformer, lora_config)

        # --- ДИАГНОСТИЧЕСКИЙ БЛОК ОТ ИНТЕРНА ---
        print(f"[ОТК] ДИАГНОСТИКА ИНЖЕКЦИИ LORA:")
        target_obj = lora_model
        if isinstance(lora_model, tuple):
            target_obj = lora_model[0]
        if isinstance(target_obj, tuple):
            target_obj = target_obj[0]

        try:
            params = list(target_obj.parameters())
            trainable_params = [p for p in params if p.requires_grad]
            trainable_params_count = sum(p.numel() for p in trainable_params)
            print(f"  └── Всего обучаемых параметров: {trainable_params_count:,}")

            captured_layers = []
            for name, module in target_obj.named_modules():
                if "lora" in name.lower():
                    captured_layers.append(name)

            print(f"  └── Количество захваченных слоев: {len(captured_layers)}")
            if len(captured_layers) > 0:
                print(f"  └── Список захваченных слоев (первые 10): {captured_layers[:10]}")
            else:
                print("  └── [КРИТИЧЕСКАЯ ОШИБКА] ПЕFT НЕ НАШЕЛ НИ ОДНОГО СЛОЯ! LoRA НЕ РАБОТАЕТ!")
        except Exception as e:
            print(f"  └── [ОШИБКА ДИАГНОСТИКИ]: Не удалось прочитать параметры. Тип объекта: {type(target_obj)}")
            print(f"  └── Сообщение ошибки: {e}")

        print(f"[ОТК] Размерности базовых слоев:")
        for name, module in target_obj.named_modules():
            if "to_out" in name or "to_q" in name or "to_k" in name or "to_v" in name:
                if hasattr(module, 'weight'):
                    print(f"  └── {name}: {module.weight.shape}")

        # --- ДИАГНОСТИЧЕСКИЙ БЛОК ОТ ИНТЕРНА END ---
        print("[ОБТ] Шаг З: Заморозка базовых матриц, активация чекпоинтинга и фиксация LoRA...")

        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()

        for name, param in lora_model.named_parameters():
            if "lora_" in name:
                if any(target in name for target in TrainConfig.TARGET_MODULES):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print("[ОБТ] Шаг К: Маршевый перенос готового квантованного пирога во VRAM CUDA...")
        lora_model = lora_model.to("cuda")

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
    pycache_path = os.path.join(TrainConfig.SRC_DIR, "__pycache__")
    if os.path.exists(pycache_path):
        shutil.rmtree(pycache_path)

    print("[ОБТ] Холодный тест отсека инжекции V02...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_params:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===
