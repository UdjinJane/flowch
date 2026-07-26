import os
import json
import gc
import torch
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel
from peft import get_peft_model, LoraConfig
from config import TrainConfig
from torchao.quantization import quantize_, int8_weight_only

# Класс FakeConfig внедрен, параметры Chroma1 настроены: 19 слоев, head_dim 128, in_channels 64
class FakeConfig:
    def __init__(self):
        self.attention_head_dim = 128
        self.guidance_embeds = False
        self.in_channels = 64
        self.joint_attention_dim = 4096
        self.num_attention_heads = 24
        self.num_layers = 19
        self.num_single_layers = 38
        self.patch_size = 1

class FluxLoraCoreV02:

    @staticmethod

    def init_transformer_with_lora():
        # Загрузка и первичная инициализация (аналогично логам)
        with open(os.path.join(TrainConfig.SRC_DIR, "transformer_config.json"), "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)
        transformer = FluxTransformer2DModel.from_config(config_dict).to(dtype=torch.bfloat16)

        # Загрузка весов и применение броневого листа конфигурации
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        clean_state_dict = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items()}
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer.config = FakeConfig() # Внедряем FakeConfig

        # Квантование и донастройка (TorchAO, int8_weight_only)
        quantize_(transformer, int8_weight_only())

        # Чистка VRAM
        torch.cuda.empty_cache()
        gc.collect()

        # Изоляция эмбеддеров в bf16
        for attr in ["x_embedder", "time_text_embed", "context_embedder"]:
            if hasattr(transformer, attr):
                setattr(transformer, attr, getattr(transformer, attr).to(dtype=torch.bfloat16))
        # Конфигурируем PEFT LoRA и внедряем в квантованное ядро [1.10]
        lora_config = LoraConfig(r=TrainConfig.LORA_RANK, lora_alpha=TrainConfig.LORA_ALPHA, target_modules=list(TrainConfig.TARGET_MODULES), bias="none")
        # Оборачиваем трансформер в полноценную структуру PEFT
        model = get_peft_model(transformer, lora_config)
        
        # 7. Гарантированная блокировка базового ядра и активация автограда только для LoRA
        model.base_model.mapping.requires_grad_(False)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model.to("cuda")

# === БЛОК 4: ХОЛОДНЫЙ ТЕСТ И МОНИТОРИНГ VRAM ===
if __name__ == "__main__":
    # Инициализация, проверка параметров и замер потребления VRAM [1.10]
    tested_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in tested_model.parameters() if p.requires_grad)
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Активных LoRA мишеней: {trainable_params:,} | VRAM: {allocated:.2f} GB")
