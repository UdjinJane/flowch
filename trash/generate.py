import os
import torch
from PIL import Image
import numpy as np
from .config import OUTPUT_DIR

def run_inference(loaded_transformer=None, epoch=0, text_embedding=None, steps=25, device='cuda'):
    """
    Безотходный контур рендеринга с поддержкой текстового кондишенирования и изоляцией RNG.
    """
    old_rng_state = torch.get_rng_state()
    old_cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    
    transformer = loaded_transformer
    was_training = transformer.training if transformer is not None else False

    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        if transformer is not None:
            transformer.eval()

        # Каноничный шейп латентов Chroma1-HD (64 канала)
        x_t = torch.randn(1, 64, 64, 64, device=device)
        dt = 1.0 / steps
        print(f"🔄 Запуск ODE траектории Rectified Flow ({steps} шагов)...")
        
        with torch.no_grad():
            for i in range(steps):
                t_curr = i * dt
                t_tensor = torch.ones(1, device=device) * t_curr
                
                if text_embedding is not None:
                    cond = text_embedding.to(device)
                    velocity = transformer(x_t, t_tensor, cond)
                else:
                    velocity = transformer(x_t, t_tensor)
                    
                x_t = x_t + velocity * dt
                if (i + 1) % 5 == 0 or (i + 1) == steps:
                    print(f"  [~] Прогресс ODE: {int(((i + 1) / steps) * 100)}%")

        print("💾 Траектория завершена! Кастинг матрицы в RGB...")
        
        # Алгоритм визуализации латентов без VAE-декодера
        x_t_vis = x_t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        x_t_vis = (x_t_vis - x_t_vis.min()) / (x_t_vis.max() - x_t_vis.min() + 1e-5)
        
        img_tensor = torch.nn.functional.interpolate(x_t_vis, size=(512, 512), mode='bilinear')
        img_array = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        final_img = Image.fromarray(img_array)

        # Безопасный абсолютный путь к отсеку рендеров
        output_dir = r"Z:\flowch\output\images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"mng_render_epoch_{epoch}.png")
        final_img.save(output_path)
        print(f"🎉 Рендеринг успешно завершен! Файл: {output_path}")


    finally:
        # Гарантированный возврат тренера в рабочий режим обучения
        if transformer is not None and was_training:
            transformer.train()
            
        torch.set_rng_state(old_rng_state)
        if old_cuda_rng_state is not None:
            torch.cuda.set_rng_state(old_cuda_rng_state)
