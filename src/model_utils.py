import torch
import torch.nn as nn
import math

class ChromaLoraWrapper(nn.Module):
    """
    астомная обертка для FP8-слоев Chroma1-HD.
    Сохраняет оригинальный FP8 слой замороженным и добавляет обучаемый BF16 LoRA-путь на нужном девайсе.
    """
    def __init__(self, original_layer, rank=16, alpha=32, scale_weight=1.0, scale_input=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        self.scale_weight = scale_weight
        self.scale_input = scale_input

        if hasattr(original_layer, 'weight'):
            out_features, in_features = original_layer.weight.shape
            target_device = original_layer.weight.device
        else:
            raise AttributeError("ригинальный слой не содержит матрицу весов (weight)")

        # нициализируем LoRA-матрицы строго на том же девайсе, что и базовая модель
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank), dtype=torch.bfloat16, device=target_device))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features), dtype=torch.bfloat16, device=target_device))
        
        # OСЯ Я С Т Т Т
        # атрицу  заполняем легким нормальным шумом
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(rank))
        # Ш: место абсолютного нуля инициализируем lora_B ультра-малым шумом,
        # чтобы открыть ворота для градиентов PyTorch обратного прохода!
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-5)

    def forward(self, x):
        base_output = self.original_layer(x)
        x_bf16 = x.to(torch.bfloat16)
        lora_output = (x_bf16 @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output.to(base_output.dtype)

def inject_chroma_lora(model, target_rank=16, target_alpha=32):
    """
    втоматический обход архитектуры Chroma1-HD и замена целевых линейных слоев на LoRA-обертки.
    """
    injected_count = 0
    
    for name, module in model.named_modules():
        is_target_layer = any(x in name for x in [
            "img_attn.qkv",      
            "img_attn.proj",     
            "linear1",           
            "linear2"            
        ])
        
        if is_target_layer and hasattr(module, 'weight'):
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name) if parent_name else model
            
            sw = getattr(module, 'scale_weight', 1.0)
            si = getattr(module, 'scale_input', 1.0)
            
            module.weight.requires_grad = False
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = False
                
            wrapper = ChromaLoraWrapper(
                original_layer=module,
                rank=target_rank,
                alpha=target_alpha,
                scale_weight=sw,
                scale_input=si
            )
            
            setattr(parent_module, layer_name, wrapper)
            injected_count += 1
            
    print(f"🎯 одификация графа завершена! спешно внедрено {injected_count} LoRA-модулей в архитектуру Chroma1-HD.")
    return model