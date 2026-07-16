# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os
import sys
import json
import torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict as peft_get_state_dict
# ... (новые импорты)

from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig
# Инжектируем нативный квантовый инструмент PyTorch Foundation
from torchao.quantization import quantize_, float8_weight_only

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        # Сборка и загрузка модели
        config_json_path = os.path.join(os.path.dirname(__file__), "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        
        transformer = FluxTransformer2DModel.from_config(config_dict)
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очистка ключей и загрузка весов
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v 
            for k, v in state_dict.items()
        }
        transformer.to_empty(device="cuda")
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer.to(torch.bfloat16)

        # Инжекция LoRA с временной изоляцией старой версии torchao от PEFT
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK, 
            lora_alpha=TrainConfig.LORA_ALPHA, 
            target_modules=TrainConfig.TARGET_MODULES, 
            bias="none"
        )
        
        # Тотальный перехват: выжигаем проверку прямо в модулях диспетчера PEFT
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        
        orig_lora_ao = getattr(peft.tuners.lora.torchao, "is_torchao_available", None)
        orig_tune_ao = getattr(peft.tuners.tuners_utils, "is_torchao_available", None)
        
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False

        try:
            lora_model = get_peft_model(transformer, lora_config)
        finally:
            # Безусловное восстановление магистралей
            if orig_lora_ao is not None: peft.tuners.lora.torchao.is_torchao_available = orig_lora_ao
            if orig_tune_ao is not None: peft.tuners.tuners_utils.is_torchao_available = orig_tune_ao



        # Квантование TorchAO FP8 (за исключением x_embedder и lora_)
        def filter_fn(mod, name):
            return not ("x_embedder" in name or "lora_" in name) and isinstance(mod, torch.nn.Linear)
        quantize_(lora_model, float8_weight_only(), filter_fn)

        # Фиксация и финализация
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()
        lora_model.to(device="cuda").to(torch.bfloat16)
        
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
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    print("[ОБТ] Холодный тест отсека инжекции V02...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"--- [ОТК] ТЕСТ ИНЖЕКЦИИ LORA V02 ПРОЙДЕН (Активные веса: {trainable_params:,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===

