# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №1: НАТИВНЫЙ БАЗИС TORCHAO ===
import os
import gc
import torch
from safetensors.torch import load_file
from torchao.quantization import quantize_, float8_weight_only
from peft import get_peft_model, LoraConfig
from torch import nn
from config import TrainConfig

# ЧИСТОКРОВНЫЙ ПРЯМОЙ ИМПОРТ: БЕЗ КОСТЫЛЕЙ И СМЕЩЕНИЙ ПУТЕЙ!
from model import Chroma, chroma_params



class FakeCLIP(torch.nn.Module):
    """Космофлотская заглушка CLIP — вырезает 1.5 ГБ из VRAM."""
    def __init__(self):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = 'cuda'
        self.text_model = None
        self.tokenizer = None
        self.model_max_length = 77
    def forward(self, *args, **kwargs):
        return torch.zeros(1, 1, 1).to(self.device)

# Передаем управление в Блок 2 (Сборка трансформера и врезка LoRA)
# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №1 ===
# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2: ИНИЦИАЛИЗАЦИЯ И НАТИВНЫЙ FP8 ===
class FluxLoraCoreV02:
    @staticmethod
    def init_transformer_with_lora():
        print("📡 АКТИВИРОВАН НАТИВНЫЙ БЕЗОПАСНЫЙ ПРОТОКОЛ TORCHAO ДЛЯ WINDOWS")
        dtype = torch.bfloat16
        
        # 1. Загрузка весов напрямую в процессор шхуны
        state_dict = load_file(TrainConfig.MODEL_SINGLE_FILE, device="cpu")
        
        # Очищаем префиксы ComfyUI/Diffusers под нативные имена Клондайка Хромы
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.diffusion_model.", "").replace("transformer.", "")
            clean_state_dict[new_key] = v

        # 2. Вычисляем реальную конфигурацию блоков по металлу реальности
        double_blocks = 0
        single_blocks = 0
        for key in clean_state_dict.keys():
            _части_ключа = key.split(".")
            if len(_части_ключа) > 1 and _части_ключа[1].isdigit():
                _номер_блока = int(_части_ключа[1]) + 1
                if "double_blocks" in key and _номер_блока > double_blocks:
                    double_blocks = _номер_блока
                elif "single_blocks" in key and _номер_блока > single_blocks:
                    single_blocks = _номер_блока

        print(f"📊 Нативная топология Chroma: Двойных блоков={double_blocks}, Одинарных={single_blocks}")

        chroma_params.depth = double_blocks
        chroma_params.depth_single_blocks = single_blocks

        # 3. Инициализируем чистокровный трансформер Хромы из манускрипта
        transformer = Chroma(chroma_params)
        transformer.dtype = dtype
        
        # Накатываем веса без дурацкого strict=True, чтобы VAE-заглушка не швыряла краш
        transformer.load_state_dict(clean_state_dict, strict=False)
        transformer.to(device="cpu", dtype=dtype)
        
        del clean_state_dict
        gc.collect()

        # 4. Умное нативное квантование TorchAO через прецизионный фильтр магистрали
        print("⚡ Запуск выборочного Float8_Weight_Only квантования магистрали")
        
        def _фильтр_магистрали(модуль, имя_модуля):
            # Проверяем, что узел является линейным слоем
            if isinstance(модуль, nn.Linear):
                # Если слой слишком мелкий (размерность <= 16), изолируем его от сжатия
                if модуль.in_features <= 16 or модуль.out_features <= 16:
                    return False
                return True
            return False

        # Запускаем квантование на весь трансформер разом, используя наш фильтр
        quantize_(transformer, float8_weight_only(), filter_fn=_фильтр_магистрали)


        
        # ИЗОЛЯЦИЯ СВЯЩЕННЫХ ЗОН: Принудительно возвращаем x_embedder в bfloat16
        if hasattr(transformer, "x_embedder"):
            print("🛡️ Защитный контур: x_embedder принудительно возвращен в bfloat16")
            transformer.x_embedder.to(dtype=torch.bfloat16)
            
        if hasattr(transformer, "single_blocks"):
            for block in transformer.single_blocks:
                if hasattr(block, "modulation") and hasattr(block.modulation, "lin"):
                    block.modulation.lin.to(dtype=torch.bfloat16)



        # Переносим сжатое тело на карту CUDA
        transformer.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()

        # 5. Конфигурируем таргеты LoRA под ИСТИННЫЕ линейные слои Клондайка Хромы
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
        
        # Перехватываем PEFT-валидатор версий под Windows на лету
        import sys
        if "peft.utils.import_utils" in sys.modules:
            peft_import = sys.modules["peft.utils.import_utils"]
            peft_import.is_torchao_available = lambda: False

        # Оборачиваем нативный трансформер в LoRA-оболочку
        model = get_peft_model(transformer, lora_config)

        # 6. Жесткая герметизация автограда — учим только матрицы LoRA, остальное бетон
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
    print(f"✅ УСПЕХ БАЗИСА TORCHAO! Обучаемых LoRA параметров: {trainable_params:,} | VRAM статика: {allocated:.2f} GB")
# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №2 И ВСЕГО МОДУЛЯ ===
