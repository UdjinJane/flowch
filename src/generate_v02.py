# Финальная версия generate_v02.py с примененными правками:

import os
import torch
import time
from PIL import Image
from config import TrainConfig
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from model_runner_v02 import run_lora_model_step

def run_inference_v02(loaded_transformer=None, current_step=0, text_embedding=None, steps=25, device='cuda'):
    """
    [МАРШРУТ V02] Изолированный рендеринг кадра.
    """
    if loaded_transformer is None:
        print("[ОТК] Ошибка: Трансформер не передан!")
        return

    print(f"\n[ОБТ] >>> Бортовой рендеринг V02 | Шаг #{current_step} <<<")
    t_start = time.time()

    print("[ОБТ] Фаза А: Заморозка RNG и перевод в режим .eval()...")
    old_rng = torch.get_rng_state()
    was_training = loaded_transformer.training
    torch.manual_seed(42 + current_step)
    loaded_transformer.eval()

    print("[ОБТ] Фаза Б: Аллокация маршевого шума Chroma1...")
    latent_h, latent_w = 64, 64
    total_tokens = (latent_h // 2) * (latent_w // 2)
    x_t = torch.randn(1, total_tokens, 64, device=device, dtype=torch.bfloat16)

    print("[ОБТ] Фаза В: Развёртывание 2D геометрии координат...")
    grid_h = torch.arange(32, device=device, dtype=torch.bfloat16)[:, None].repeat(1, 32)
    grid_w = torch.arange(32, device=device, dtype=torch.bfloat16)[None, :].repeat(32, 1)
    
    img_ids = torch.zeros((total_tokens, 3), device=device, dtype=torch.bfloat16)
    img_ids[:, 1], img_ids[:, 2] = grid_h.flatten(), grid_w.flatten()
    
    # Динамическая длина ID под входящий кэш эмбеддингов контекста
    txt_len = text_embedding.shape[1] if text_embedding is not None else 256
    txt_ids = torch.zeros((txt_len, 3), device=device, dtype=torch.bfloat16)
    
    pooled_projections = torch.zeros((1, 768), device=device, dtype=torch.bfloat16)

    print("[ОБТ] Фаза Г: Подготовка текстового контекста...")
    if text_embedding is not None:
        cond = text_embedding.to(device, dtype=torch.bfloat16)
        # Если батч прилетел трехмерным [1, 1, N, D], снимаем лишнюю ось
        if cond.dim() == 4 and cond.shape[0] == 1:
            cond = cond.squeeze(0)
    else:
        print("[ОБТ] Внимание: Текстовый вектор пуст! Генерируем заглушку.")
        cond = torch.zeros((1, 256, 4096), device=device, dtype=torch.bfloat16)

    print(f"[ОБТ] Фаза Д: Запуск ODE траектории ({steps} шагов Эйлера)...")
    with torch.no_grad():
        t_lines = torch.linspace(0.0, 1.0, steps + 1, device=device)
        steps_grid = t_lines / (1.0 + (1.0 - t_lines) * 0.5)
        
        for i in range(steps):
            t_curr = steps_grid[i].item()
            t_next = steps_grid[i+1].item()
            dt = t_next - t_curr
            t_tensor = torch.ones(1, device=device) * t_curr

            # Заглушка маски для маршевого раннера
            batch_stub = {"text_ids_mask": torch.ones((1, txt_len), device=device, dtype=torch.bool)}
            
            # Вычисление вектора скорости
            velocity = run_lora_model_step(
                loaded_transformer,
                batch_stub,
                x_t,
                t_tensor,
                cond,
                pooled_projections,
                txt_ids,
                img_ids
            )
            
            # Истинный флотский буравчик: распаковка выхода модели
            pred_tensor = velocity
            if isinstance(pred_tensor, (tuple, list)):
                pred_tensor = pred_tensor[0]
            if hasattr(pred_tensor, "sample"):
                pred_tensor = pred_tensor.sample
            # Снайперский срез: забираем первые 64 канала изображения
            pred_latents = pred_tensor[:, :, :64]
            
            # Сдвиг по траектории потока (Шаг Эйлера)
            x_t = x_t + pred_latents * dt

            if (i + 1) % 5 == 0 or (i + 1) == steps:
                t_show = t_next if (i + 1) == steps else t_curr
                print(f"  [~] Траектория ODE: {int(((i + 1) / steps) * 100)}% | t = {t_show:.3f}")
                print("[ОБТ] Фаза Е: Подготовка параметров Flux VAE...")
                v_conf = {
                    "_class_name": "AutoencoderKL",
                    "block_out_channels": (128, 256, 512, 512),
                    "latent_channels": 16,
                    "scaling_factor": 0.3611,
                    "shift_factor": 0.1159,
                    # Остальные параметры инициализации
                }
                vae = AutoencoderKL.from_config(v_conf)

                print("[ОБТ] Загрузка весов VAE и перевод контура в bfloat16...")
                vae_state = load_file(TrainConfig.VAE_PATH, device="cpu")
                vae_clean = {k.replace("vae.", "") if k.startswith("vae.") else k: v for k, v in vae_state.items()}
                vae.load_state_dict(vae_clean, strict=True)
                vae = vae.to(device=device, dtype=torch.bfloat16)

                print("[ОБТ] Фаза Ж: Распаковка 2D патчей и маршевый VAE-декод...")
                with torch.no_grad():
                    b_sz = x_t.shape[0]
                    # Из 1024 токенов восстанавливаем 2D сетку патчей [B, 32, 32, 16, 2, 2]
                    latents_4d = x_t.view(b_sz, 32, 32, 16, 2, 2)
                    # Собираем пространственные оси обратно в шейп Flux [B, 16, 64, 64]
                    latents_4d = latents_4d.permute(0, 3, 1, 4, 2, 5).reshape(b_sz, 16, 64, 64)
                    
                    # Исправленная денормализация Flux: снайперски снимаем сдвиг и масштаб
                    latents_decoded = (latents_4d * v_conf["scaling_factor"]) + v_conf["shift_factor"]
                    
                    rgb_tensor = vae.decode(latents_decoded.to(device, dtype=torch.bfloat16), return_dict=False)[0]

    # Извлекаем чистый тензор из кортежа вывода VAE
    if isinstance(rgb_tensor, (tuple, list)):
        rgb_tensor = rgb_tensor[0]

    # Нормализация пикселей в диапазон [0, 1] и перевод в формат изображения
    rgb_tensor = (rgb_tensor / 2 + 0.5).clamp(0, 1)
    img_array = (rgb_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype('uint8')
    
    output_dir = os.path.join(TrainConfig.OUTPUT_DIR, "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mng_render_step_{current_step}.png")
    Image.fromarray(img_array).save(output_path)

    # Безусловное выжигание VAE из памяти для удержания лимита VRAM < 21.0 GB
    del vae, vae_state, vae_clean, latents_4d, latents_decoded, rgb_tensor
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("[ОБТ] Фаза З: Деактивация контура инференса, возврат флагов...")
    if was_training:
        loaded_transformer.train()
    torch.set_rng_state(old_rng)
    
    t_end = time.time()
    print(f"[УСПЕХ] Рендеринг выполнен за {t_end - t_start:.2f} s. Файл зафиксирован: {output_path}\n")