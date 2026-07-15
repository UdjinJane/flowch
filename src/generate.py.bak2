import os
import torch
import argparse
from PIL import Image
import numpy as np
from safetensors.torch import load_file
from .config import VAE_PATH, OUTPUT_DIR, device

def run_inference(loaded_transformer=None, epoch=0):
    # 1. сли модель передана напрямую из тренера — используем её без клонирования!
    if loaded_transformer is not None:
        print(f"📸 [Сэмплер] спользуем прогретый Transformer из памяти тренера...")
        transformer = loaded_transformer
    else:
        # огика для ручного запуска из консоли (через аргументы)
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, required=True)
        parser.add_argument('--epoch', type=int, default=0)
        args = parser.parse_args()
        epoch = args.epoch

        print(f"🚀 агрузка чекпоинта LoRA: {args.checkpoint}")
        from .models import EmptyTransformer
        from .model_utils import inject_chroma_lora
        
        transformer = EmptyTransformer().to(device)
        transformer = inject_chroma_lora(transformer)
        
        if os.path.exists(args.checkpoint):
            lora_sd = load_file(args.checkpoint)
            transformer.load_state_dict(lora_sd, strict=False)
    
    # апоминаем исходный режим (train/eval), чтобы вернуть модель тренеру в неизменном виде
    was_training = transformer.training
    transformer.eval()

    # аноничный стартовый шум (дисперсия = 1.0)
    torch.manual_seed(42)
    x_t = torch.randn(1, 64, 64, 64, device=device) 
    steps = 25
    dt = 1.0 / steps

    print(f"🔄 апуск ODE траектории Rectified Flow ({steps} шагов)...")
    with torch.no_grad():
        for i in range(steps):
            t_curr = i * dt
            t_tensor = torch.ones(1, device=device) * t_curr
            velocity = transformer(x_t, t_tensor)
            x_t = x_t + velocity * dt
            if (i+1) % 5 == 0 or i == steps - 1:
                print(f"  [~] рогресс ODE: {int((i + 1) / steps * 100)}%")

    print("💾 Траектория завершена! астинг матрицы в RGB...")
    x_t = x_t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    x_t = (x_t - x_t.min()) / (x_t.max() - x_t.min() + 1e-5)
    
    img_tensor = torch.nn.functional.interpolate(x_t, size=(512, 512), mode='bilinear')
    img_array = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    final_img = Image.fromarray(img_array)

    output_path = os.path.join(OUTPUT_DIR, "output", "images", f"mng_render_epoch_{epoch}.png")
    final_img.save(output_path)
    print(f"🎉 ендеринг успешно завершен! айл: {output_path}")
    
    # озвращаем модель в исходное состояние для продолжения плавки
    if was_training:
        transformer.train()

if __name__ == '__main__':
    run_inference()
