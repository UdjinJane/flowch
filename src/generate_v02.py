import os
import torch
import time
from PIL import Image
from config import TrainConfig

def run_inference_v02(loaded_transformer=None, current_step=0, text_embedding=None, steps=25, device='cuda'):
    """
    Изолированный контур рендеринга V02 (Chroma1, 2D патчи, RNG-контроль, полная телеметрия).
    """
    if loaded_transformer is None:
        print("[ОТК] Ошибка рендеринга V02: Трансформер не передан в контур инференса!")
        return
    
    print(f"\n[ОБТ] >>> Запуск бортового рендеринга V02 на шаге обучения # {current_step} <<<")
    t_start = time.time()
    
    print("[ОБТ] Фаза А: Изоляция RNG-генератора и фиксация состояния .eval()...")
    old_rng = torch.get_rng_state()
    was_training = loaded_transformer.training
    torch.manual_seed(42)
    loaded_transformer.eval()

    print("[ОБТ] Фаза Б: Аллокация маршевого шума Chroma1 (1, 1024 токена, 64 канала)...")
    latent_h, latent_w = 64, 64
    total_tokens = (latent_h // 2) * (latent_w // 2)
    x_t = torch.randn(1, total_tokens, 64, device=device, dtype=torch.bfloat16)
    
    print("[ОБТ] Фаза В: Развёртывание 2D геометрии img_ids и заглушек txt_ids...")
    grid_h = torch.arange(32, device=device)[:, None].repeat(1, 32)
    grid_w = torch.arange(32, device=device)[None, :].repeat(32, 1)
    img_ids = torch.zeros(total_tokens, 3, device=device, dtype=torch.bfloat16)
    img_ids[:, 1], img_ids[:, 2] = grid_h.flatten(), grid_w.flatten()
    txt_ids = torch.zeros(256, 3, device=device, dtype=torch.bfloat16)
    
    # Добавляем обязательный пулинг для эмбеддера времени
    pooled_projections = torch.zeros(1, 768, device=device, dtype=torch.bfloat16)

    print("[ОБТ] Фаза Г: Кастинг текстовых эмбеддингов Т5 в bfloat16...")
    if text_embedding is not None:
        cond = text_embedding.to(device, dtype=torch.bfloat16)
    else:
        print("[ОБТ] Внимание: Текстовый вектор пуст! Генерируем нулевой контекст [1, 256, 4096].")
        cond = torch.zeros(1, 256, 4096, device=device, dtype=torch.bfloat16)
    
    print(f"[ОБТ] Фаза Д: Запуск ODE траектории Rectified Flow ({steps} шагов Эйлера)...")
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_curr = i * dt
            t_tensor = torch.ones(1, device=device) * t_curr
            
            # Маршевый проход через трансформер с полной стыковкой эмбеддингов
            velocity = loaded_transformer(
                hidden_states=x_t, 
                timestep=t_tensor, 
                encoder_hidden_states=cond, 
                pooled_projections=pooled_projections,
                img_ids=img_ids, 
                txt_ids=txt_ids,
                return_dict=False
            )

            # Истинный флотский буравчик: снайперски выбивает нулевой элемент до чистого тензора
            pred_tensor = velocity
            while isinstance(pred_tensor, (tuple, list)):
                pred_tensor = pred_tensor[0]

            # Если это объект diffusers output, забираем его sample
            if hasattr(pred_tensor, "sample"):
                pred_tensor = pred_tensor.sample
            
            # Снайперский флотский срез: забираем строго первые 64 канала изображения, отсекая текстовый мусор
            pred_latents = pred_tensor[:, :, :64]

            # Шаг по траектории потока изображения
            x_t = x_t + pred_latents * dt

            
            if (i + 1) % 5 == 0 or (i + 1) == steps:
                print(f" [~] Траектория ODE: {int(((i + 1) / steps) * 100)}% | Текущий t = {t_curr:.3f}")

    print("[ОБТ] Фаза Е: Траектория завершена! Схлопывание 2D патчей в ЧБ-матрицу...")
    # Преобразуем плоские 1024 токена обратно в геометрию кадра 32х32 для экспресс-визуализации
    x_t_vis = x_t.mean(dim=-1).view(1, 1, 32, 32).repeat(1, 3, 1, 1)
    x_t_vis = (x_t_vis - x_t_vis.min()) / (x_t_vis.max() - x_t_vis.min() + 1e-5)
    
    print("[ОБТ] Фаза Ж: Интерполяция матрицы до 512px и сохранение на SSD...")
    img_tensor = torch.nn.functional.interpolate(x_t_vis, size=(512, 512), mode='bilinear')
    # Принудительно сбрасываем bfloat16 в float32 перед отправкой в numpy массив
    img_array = (img_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype('uint8')

    
    output_dir = os.path.join(TrainConfig.ROOT_DIR, "output", "images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mng_render_step_{current_step}.png")
    
    Image.fromarray(img_array).save(output_path)
    
    print("[ОБТ] Фаза З: Деактивация контура инференса, возврат флагов .train()...")
    if was_training: 
        loaded_transformer.train()
    torch.set_rng_state(old_rng)
    
    t_end = time.time()
    print(f"[УСПЕХ] Рендеринг запекаемой эпохи выполнен за {t_end - t_start:.2f}s. Файл зафиксирован: {output_path}\n")
