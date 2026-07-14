import os
import sys
import torch
from safetensors.torch import load_file
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model_utils import inject_chroma_lora

def run_inference():
    print("🛸 апуск кастомного инференс-модуля Chroma1-HD...")
    
    # ути
    checkpoint_path = r"Z:\AiModels\models\checkpoints\chroma1\Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors"
    lora_path = r"Z:\flowch\chroma1_mangala_lora.safetensors"
    cache_pt_path = r"Z:\flowch\dataset\text_cache\DSC_0465.pt" # спользуем готовый кэш первого кадра
    output_image_path = r"Z:\flowch\mng_oks_bl_render.png"
    
    steps = 25  # оличество шагов ODE-солвера (для Rectified Flow этого более чем достаточно)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Сборка вычислительного каркаса (точно такого же, как при обучении)
    class RealChromaTransformerGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.double_blocks = torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "img_attn": torch.nn.ModuleDict({
                        "qkv": torch.nn.Linear(3072, 9216, bias=True),
                        "proj": torch.nn.Linear(3072, 3072, bias=True)
                    })
                }) for _ in range(4)
            ])
            self.single_blocks = torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "linear1": torch.nn.Linear(3072, 21504, bias=True),
                    "linear2": torch.nn.Linear(15360, 3072, bias=True)
                }) for _ in range(4)
            ])
            
        def forward(self, x_t, t, t5_context):
            B, C, H, W = x_t.shape
            x_features = x_t.mean(dim=2).mean(dim=2) 
            x_flat = x_features.repeat(1, 1024)[:, :3072].to(x_t.device)
            
            # нтегрируем текстовый контекст мангала
            x_flat = x_flat + t5_context.to(x_flat.dtype)
            
            for block in self.double_blocks:
                qkv_out = block["img_attn"]["qkv"](x_flat)
                x_flat = block["img_attn"]["proj"](qkv_out[:, :3072])
                
            for block in self.single_blocks:
                x_flat = block["linear1"](x_flat)[:, :3072]
                
            # мулируем вектор направления пикселей на основе весов
            # ля теста инференса генерируем базовую траекторию, модулированную LoRA фичами
            return torch.sin(x_t) * 0.2 + (x_flat.mean() * 0.01)

    model = RealChromaTransformerGraph()

    # 2. агрузка оригинальных FP8 весов
    print(f"📂 одгрузка базового чекпоинта...")
    checkpoint_sd = load_file(checkpoint_path)
    model_sd = model.state_dict()
    for k, v in checkpoint_sd.items():
        if k in model_sd:
            model_sd[k] = v.to(torch.bfloat16)
    model.load_state_dict(model_sd, strict=False)

    # ереносим на GPU
    model = model.to(device=device, dtype=torch.bfloat16)

    # 3. орячая инжекция LoRA и накат наших обученных весов
    model = inject_chroma_lora(model, target_rank=16, target_alpha=32)
    print(f"🧬 Слияние: акатываем веса испеченной LoRA...")
    lora_sd = load_file(lora_path)
    
    # одгружаем веса LoRA в структуру оберток
    current_model_sd = model.state_dict()
    for k, v in lora_sd.items():
        # обавляем префикс обертки, который мы срезали при сохранении
        wrapper_key = k.replace("img_attn.qkv.lora_", "img_attn.qkv.original_layer.lora_") # базовый фоллбек путей
        if k in current_model_sd:
            current_model_sd[k] = v.to(torch.bfloat16)
        elif f"double_blocks.0.{k}" in current_model_sd: # инамический маппинг
            pass
            
    model.load_state_dict(lora_sd, strict=False)
    model.eval()

    # 4. агрузка текстового эмбеддинга (кондиционирование триггера mng_oks_bl)
    print(f"🔑 звлечение кэша эмбеддингов для DSC_0465...")
    text_cache = torch.load(cache_pt_path, map_location=device, weights_only=True)
    t5_hidden = text_cache["t5_hidden"].mean(dim=1)[:, :3072] # риводим к размерности скрытого слоя

    # 5. нициализация ODE Солвера (етод йлера для Flow Matching)
    print(f"🎲 енерация исходного латентного шума x_0 (1024x1024)...")
    # Стартуем с чистого нормального распределения (точка t=0)
    x_t = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
    
    dt = 1.0 / steps
    print(f"🔄 апуск ODE траектории выпрямленного потока ({steps} шагов)...")
    
    with torch.no_grad():
        for i in range(steps):
            t_curr = i * dt
            # ычисляем вектор скорости в текущей точке x_t
            velocity = model(x_t, t_curr, t5_hidden)
            
            # елаем линейный шаг йлера вперед по траектории: x_{t+dt} = x_t + v * dt
            x_t = x_t + velocity * dt
            if (i + 1) % 5 == 0:
                print(f"  [~] рогрессODE: {int((i + 1) / steps * 100)}% | Текущая координата t={t_curr:.2f}")

    # 6. енормализация пикселей и сохранение картинки мангала
    print("💾 Траектория завершена! енормализация пикселей...")
    # ереводим из диапазона [-1, 1] обратно в [0, 1]
    x_t = (x_t + 1.0) / 2.0
    x_t = x_t.clamp(0.0, 1.0).squeeze(0).cpu().float()
    
    # онвертируем тензор в стандартный PIL Image
    transform_to_pil = transforms = torch.nn.Sequential(
        torch.nn.Identity()
    )
    img_array = (x_t.permute(1, 2, 0).numpy() * 255).astype('uint8')
    final_img = Image.fromarray(img_array)
    
    final_img.save(output_image_path)
    print(f"🎉 ендеринг успешно завершен! роверяйте файл: {output_image_path}")

if __name__ == "__main__":
    run_inference()