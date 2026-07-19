# === БЛОК ЯДРА LORA V02 СТАРТ ===
import os, json, time, torch
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("[ОБТ] Магистральный запуск инжектора: lora_core_v02")

        # === БЛОК 2: БЕЗОПАСНОЕ ЧТЕНИЕ И КАСТ В BF16 ===
        # Считывание конфигурации ядра с подавлением UTF-8 BOM маркера
        config_json_path = os.path.join(TrainConfig.SRC_DIR, "transformer_config.json")
        with open(config_json_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        
        # Сборка чистого каркаса в ОЗУ
        transformer = FluxTransformer2DModel.from_config(config_dict)
        
        # Вычитка монолита весов с диска напрямую в CPU
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очистка префиксов ключей ComfyUI и принудительный каст тензоров в bfloat16
        clean_state_dict = {
            k.replace("model.diffusion_model.", "") if k.startswith("model.diffusion_model.") else k: v.to(torch.bfloat16)
            for k, v in state_dict.items()
        }
        
        # Заливка очищенных bfloat16 весов в созданный каркас
        transformer.load_state_dict(clean_state_dict, strict=False)
        print("[УСПЕХ] Базовая модель собрана в ОЗУ без раздутия весов.")

        # === БЛОК 3: ИНЖЕКЦИЯ LORA И ФИНАЛИЗАЦИЯ ===
        # Блокировка конфликтов внутри PEFT с библиотекой torchao
        import peft.tuners.lora.torchao
        import peft.tuners.tuners_utils
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.tuners_utils.is_torchao_available = lambda: False
        
        # Конфигурация целевых мишеней адаптера
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK, 
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=list(TrainConfig.TARGET_MODULES), 
            bias="none"
        )
        model = get_peft_model(transformer, lora_config)
        
        # Заморозка базовой сети и точечная активация градиентов LoRA
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        print("[УСПЕХ] Экономное ядро LoRA_Core_V02 герметизировано на GPU.")
        return model.to("cuda")
# === БЛОК ЯДРА LORA V02 ФИНАЛ ===
