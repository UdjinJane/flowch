# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os, json, time, torch
from safetensors.torch import load_file, save_file as st_save_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")
        
        # 1. Загрузка конфига и модели в bfloat16 (cpu)
        config_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_path, "r") as f: transformer = FluxTransformer2DModel.from_config(json.load(f))
        
        state_dict = {
            k.replace("model.diffusion_model.", ""): v.to(torch.bfloat16) 
            for k, v in load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu").items()
        }
        transformer.load_state_dict(state_dict, strict=False)
        
        # 2. Инжекция LoRA (отключаем torchao принудительно)
        import peft.tuners.lora.torchao; peft.tuners.lora.torchao.is_torchao_available = lambda: False
        
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK, lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES), bias="none"
        )
        lora_model = get_peft_model(transformer, lora_config)
        
        # 3. Фиксация весов и перенос на GPU
        for name, param in lora_model.named_parameters():
            if "lora_" in name: param.data = param.data.to(torch.bfloat16); param.requires_grad = True
            else: param.requires_grad = False
        
        return lora_model.to("cuda")

if __name__ == "__main__":
    print("[ОБТ] Холодный тест V02...")
    model_test = FluxLoraCoreV02.init_transformer_with_lora()
    print(f"--- [ОТК] ТЕСТ ПРОЙДЕН (Params: {sum(p.numel() for p in model_test.parameters() if p.requires_grad):,}) ---")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===
