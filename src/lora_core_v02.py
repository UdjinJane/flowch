# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os, json, torch
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")

        # Загрузка конфига и базы в bfloat16
        with open(os.path.join(TrainConfig.SRC_DIR, "transformer_config.json"), "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
            
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.bfloat16)
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очистка ключей и приведение типов
        clean_state_dict = {
            k.replace("model.diffusion_model.", ""): v.to(torch.bfloat16)
            for k, v in state_dict.items()
        }
        transformer.load_state_dict(clean_state_dict, strict=False)
        print("[УСПЕХ] Базовая модель герметизирована.")

        # Блокировка конфликтов torchao
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        
        lora_config = LoraConfig(r=TrainConfig.LORA_RANK, target_modules=list(TrainConfig.TARGET_MODULES), bias="none")
        model = get_peft_model(transformer, lora_config)
        
        # Заморозка базы
        for name, param in model.named_parameters():
            if "lora_" not in name: param.requires_grad = False
                
        return model.to("cuda")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===

