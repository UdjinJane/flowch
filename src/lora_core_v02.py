# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict as peft_get_state_dict
from config import TrainConfig
from torchao.quantization import quantize_, float8_weight_only



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
        # Превращаем список слоев в строгое регулярное выражение для PEFT
        # Пример: '.*(to_q\.0|to_out\.0)$' гарантирует инжекцию только в указанные узлы
        import re
        # Сначала спокойно экранируем точки в обычном массиве строк
        # Отрезаем '.0' и строим жесткую регулярку конца имени для PEFT.
        # Результат для hard_object: '.*\\.(to_q|to_out)$'
        clean_targets = [t.replace('.0', '') for t in TrainConfig.TARGET_MODULES]
        target_regex = f".*\\.({'|'.join(clean_targets)})$"


        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=target_regex,
            bias="none"
        )


        
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        orig_lora_ao = getattr(peft.tuners.lora.torchao, "is_torchao_available", None)
        orig_tune_ao = getattr(peft.tuners.tuners_utils, "is_torchao_available", None)
        
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False

        try:
            print("[ОБТ] Шаг Ж: Инжекция LoRA-адаптеров в bf16-граф на CPU...")
            lora_model = get_peft_model(transformer, lora_config)
        finally:
            if orig_lora_ao is not None: peft.tuners.lora.torchao.is_torchao_available = orig_lora_ao
            if orig_tune_ao is not None: peft.tuners.tuners_utils.is_torchao_available = orig_tune_ao

        print("[ОБТ] Шаг З: Запуск нативного квантования TorchAO FP8 (Фильтрация слоев)...")
        # Жесткий флотский фильтр: полностью изолируем веса LoRA от квантования FP8, оставляя их в обучаемом bf16
        def filter_fn(mod, name):
            is_linear = isinstance(mod, torch.nn.Linear)
            is_base_layer = not ("x_embedder" in name or "lora" in name or "base_layer" in name)
            return is_linear and is_base_layer

        quantize_(lora_model, float8_weight_only(), filter_fn)

        print("[ОБТ] Шаг И: Заморозка базовых матриц, активация чекпоинтинга и фиксация LoRA...")
        # Замораживаем только базовую модель, оставляя адаптеры LoRA нетронутыми
        transformer.requires_grad_(False)
        
        # Сначала включаем чекпоинтинг, пусть он делает свои сбросы параметров
        transformer.enable_gradient_checkpointing()

        # И только теперь ЖЕСТКО и финально выжигаем градиенты для неактивных слоев LoRA
        for name, param in lora_model.named_parameters():
            if "lora_" in name:
                if any(target in name for target in TrainConfig.TARGET_MODULES):
                    param.requires_grad = True
                else:
                    param.requires_grad = False # Контрольный выстрел по градиентам to_k и to_v


        
        print("[ОБТ] Шаг К: Маршевый перенос готового квантованного пирога во VRAM CUDA...")
        # Явный маршевый перенос на видеокарту без использования внешних переменных
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

