import os
import gc
import torch
from safetensors.torch import load_file
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype
from toolkit.dequantize import patch_dequantization_on_save
from peft import get_peft_model, LoraConfig

from config import TrainConfig
# Импортируем истинную геометрию Хромы из нашего Клондайка
from src.model import Chroma, chroma_params

# Космофлотская заглушка CLIP — вырезает 1.5 ГБ мусорного веса из VRAM
class FakeCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = 'cuda'
        self.text_model = None
        self.tokenizer = None
        self.model_max_length = 77
    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1, 1).to(self.device)

class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("📡 АКТИВИРОВАН ПРОТОКОЛ СБОРКИ НАТИВНОГО ЯДРА CHROMA V50")
        dtype = torch.bfloat16
        
        # 1. Загрузка весов напрямую в процессор шхуны
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Зачищаем префиксы ComfyUI/Diffusers под нативные имена Клондайка
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.diffusion_model.", "").replace("transformer.", "")
            clean_state_dict[new_key] = v

        # 2. Вычисляем реальную конфигурацию блоков по металлу реальности
        double_blocks = 0
        single_blocks = 0
        for key in clean_state_dict.keys():
            if "double_blocks" in key:
                block_num = int(key.split(".")[1]) + 1
                if block_num > double_blocks:
                    double_blocks = block_num
            elif "single_blocks" in key:
                block_num = int(key.split(".")[1]) + 1
                if block_num > single_blocks:
                    single_blocks = block_num

        print(f"📊 Параметры Chroma: Двойных блоков={double_blocks}, Одинарных={single_blocks}")
        chroma_params.depth = double_blocks
        chroma_params.depth_single_blocks = single_blocks

        # 3. Инициализируем чистокровный трансформер
        transformer = Chroma(chroma_params)
        transformer.dtype = dtype
        
        # Накатываем веса без дурацкого strict=True
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer.to(device="cpu", dtype=dtype)
        
        del clean_state_dict
        gc.collect()

        # 4. Нативное квантование Optimum Quanto — защита от дедлоков Windows WDDM
        print("⚡ Запуск потокового квантования весов через Optimum Quanto")
        patch_dequantization_on_save(transformer)
        
        # Используем qint8 для базовых весов (канон Метрополии)
        qtype = get_qtype("qint8")
        quantize(transformer, weights=qtype)
        freeze(transformer)
        
        # Переносим сжатое тело на карту
        transformer.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()

        # 5. Конфигурируем таргеты LoRA под ИСТИННЫЕ линейные слои Клондайка Хромы
        # Вешаемся на общие QKV проекторы двойных блоков и базовые слои одиночных блоков
        target_modules = [
            "img_attn.qkv", "txt_attn.qkv", # Линейные блоки DoubleStream
            "linear1", "linear2"             # Линейные блоки SingleStream
        ]
        
        print(f"🎯 Врезка LoRA контура. Ранг: {TrainConfig.LORA_RANK}, Мишени: {target_modules}")
        lora_config = LoraConfig(
            r=TrainConfig.LORA_RANK,
            lora_alpha=TrainConfig.LORA_ALPHA,
            target_modules=target_modules,
            bias="none"
        )
        
        # Перехватываем PEFT-валидатор на лету
        import sys
        from types import ModuleType
        if "peft.utils.import_utils" in sys.modules:
            peft_import = sys.modules["peft.utils.import_utils"]
            peft_import.is_torchao_available = lambda: False

        # Оборачиваем нативный трансформер
        model = get_peft_model(transformer, lora_config)

        # 6. Жесткая герметизация автограда — учим только матрицы LoRA
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model

if __name__ == "__main__":
    tested_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = sum(p.numel() for p in tested_model.parameters() if p.requires_grad)
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"✅ УСПЕХ! Активных LoRA параметров: {trainable_params:,} | VRAM статика: {allocated:.2f} GB")
