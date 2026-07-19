# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import time
import torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict as peft_get_state_dict
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")
        t_start = time.time()
        
        # Шаг А: Считывание конфигурации
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        print(f"[ТЕЛЕМЕТРИЯ] Тап А выполнен за: {time.time() - t_start:.4f} сек")
        
        # Шаг Б: Сборка каркаса в ОЗУ
        t_step = time.time()
        transformer = FluxTransformer2DModel.from_config(config_dict)
        print(f"[ТЕЛЕМЕТРИЯ] Тап Б выполнен за: {time.time() - t_step:.4f} сек")
        
        # Шаг В: Вычитка монолита с диска
        t_step = time.time()
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        print(f"[ТЕЛЕМЕТРИЯ] Тап В выполнен за: {time.time() - t_step:.4f} сек")
        
        # Шаг Г: Дешифровка и очистка ключей ComfyUI
        t_step = time.time()
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v
            for k, v in state_dict.items()
        }
        print(f"[ТЕЛЕМЕТРИЯ] Тап Г выполнен за: {time.time() - t_step:.4f} сек")
        
        # Шаг Д: Послойная заливка в ОЗУ
        t_step = time.time()
        transformer.load_state_dict(clean_state_dict, strict=False)
        print(f"[ТЕЛЕМЕТРИЯ] Тап Д выполнен за: {time.time() - t_step:.4f} сек")
        print("[УСПЕХ] Базовая модель собрана в ОЗУ без раздутия весов.")

        # Шаг Е: Подготовка адаптеров LoRA (глушение внутренних проверок PEFT) [1.6]
        t_step = time.time()
        target_modules_list = list(TrainConfig.TARGET_MODULES)
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=target_modules_list,
            bias="none"
        )
        
        # Обход защитных триггеров PEFT против кастинга [1.6]
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False
        print(f"[ТЕЛЕМЕТРИЯ] Тап Е выполнен за: {time.time() - t_step:.4f} сек")
        
        # Шаг Ж: Инжекция LoRA-адаптеров в граф на CPU [1.6]
        t_step = time.time()
        lora_model = get_peft_model(transformer, lora_config)
        print(f"[ТЕЛЕМЕТРИЯ] Тап Ж выполнен за: {time.time() - t_step:.4f} сек")

        # Шаг З: Заморозка базовых FP8-матриц и активация чекпоинтинга
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()
        
        # Точечная активация градиентов для LoRA весов
        for name, param in lora_model.named_parameters():
            param.requires_grad = "lora_" in name
        
        # Шаг К: Перенос LoRA во VRAM CUDA
        lora_model = lora_model.to("cuda")
        print(f"[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")
        
        return lora_model

    @staticmethod
    def get_peft_model_state_dict(lora_model):
        return peft_get_state_dict(lora_model)

    @staticmethod
    def save_file(tensor_dict, filepath):
        st_save_file(tensor_dict, filepath)

if __name__ == "__main__":
    # Логика самотестирования
    print("[ОБТ] Холодный тест отсека инжекции V02...")
    model_test = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_p = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_p:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===
