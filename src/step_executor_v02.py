import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import TrainConfig
from model_runner_v02 import run_lora_model_step

#----------- БЛОК №1_4D_NATIVE_PACK_FIX --------------
# Файл: src/step_executor_v02.py
# Позиция: Замена тела функции pack_latents_to_patches (строки 8-12)
# Привязка: Сразу под комментарием """Каноническая упаковка латентов Хромы под 16x16 патчи Метрополии."""

def pack_latents_to_patches(latents):
    """Каноническая упаковка латентов Хромы под 16x16 патчи Метрополии."""
    b, c, h, w = latents.shape
    # КРИТИЧЕСКИЙ ФИКС: Возвращаем p = 2 (оригинальный шаг кузнецов) вместо взрывоопасного p = 16
    p = 2
    return latents.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)

#----------- КОНЕЦ БЛОКА №1_4D_NATIVE_PACK_FIX --------------
#----------- БЛОК №4_IMG_IDS_SCALE_FIX --------------
# Файл: src/step_executor_v02.py
# Позиция: Замена тела функции generate_chroma_img_ids (строки 22-28)
# Привязка: Сразу под комментарием """Генерация трехмерных RoPE идентификаторов кадра под длину Хромы."""

def generate_chroma_img_ids(height, width, device):
    """Генерация трехмерных RoPE идентификаторов кадра под длину Хромы."""
    # КРИТИЧЕСКИЙ ФИКС: Переводим генератор на нативный шаг Кузнецов p = 2 вместо старого p = 16
    p = 2
    h_patches, w_patches = height // p, width // p
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device, dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    return img_ids.view(1, -1, 3)

#----------- КОНЕЦ БЛОКА №4_IMG_IDS_SCALE_FIX --------------


# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2: МАРШЕВЫЙ ШАГ И СМАРТ-ТЕЛЕМЕТРИЯ ===
def execute_single_frame_step(mega_batch, frame_idx, device, lora_model):
    """Полный герметичный отсек маршевого шага кадра Хромы с 7-портовой стыковкой."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from config import TrainConfig
    from model_runner_v02 import run_lora_model_step

#---------- Старт блока №1_4D_NATIVE_PACK
    # 1. Загрузка оригинальных осей из датасета (4D контур восстановлен)
    latents = mega_batch["latents"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
    prompt_embeds = mega_batch["prompt_embeds"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)

    # 2. Математика Rectified Flow на нативных осях 4D-тензора
    t_raw = torch.rand(1, device=device, dtype=torch.bfloat16)
    t_attr = (3.0 * t_raw) / (1.0 + 2.0 * t_raw)
    noise = torch.randn_like(latents)
    t_model_scale = (t_attr.clone() * 1000.0).to(device=device, dtype=torch.bfloat16)
    
    # Прямой сдвиг мантиссы шума в 4D пространстве кадра
    noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise

    # 3. Нативная упаковка патчей: сжатие 4D (1, 16, 64, 64) -> 3D (1, 1024, 64)
    packed_noisy_latents = pack_latents_to_patches(noisy_latents)

    # 4. Нарезка позиционных сеток RoPE на основе упакованной геометрии (32x32 = 1024)
    img_ids = generate_chroma_img_ids(32 * 16, 32 * 16, device)
    
    # Геометрию масок и текстовых айдишников привязываем к реальной длине текста (prompt_embeds.shape[1] = 128)
    txt_len = prompt_embeds.shape[1]
    txt_ids_aligned = torch.zeros(1, txt_len, 3, device=device, dtype=torch.bfloat16)
    kwargs_mask = {"txt_mask": torch.ones((1, txt_len), device=device, dtype=torch.bfloat16)}
#--------- Окончание блока №1_4D_NATIVE_PACK


#---------- Старт блока №2_FINAL_POSITIONAL
    # 4. Градиентный чекпоинтинг: жесткое позиционное выравнивание 8 портов
    pooled_projections_dummy = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_tensor = checkpoint(
            run_lora_model_step,      # Назначение контура
            lora_model,                # 1. lora_model
            kwargs_mask,               # 2. batch
            packed_noisy_latents,      # 3. packed_noisy_latents (ЗДЕСЬ ДОЛЖЕН БЫТЬ КАДР, А НЕ ТЕКСТ)
            t_model_scale,             # 4. timesteps_attr
            prompt_embeds,             # 5. prompt_embeds
            pooled_projections_dummy,  # 6. pooled_projections
            txt_ids_aligned,           # 7. txt_ids
            img_ids,                   # 8. img_ids
            use_reentrant=False
        )
#--------- Окончание блока №2_FINAL_POSITIONAL

        
#---------- Старт блока №3_PATCH
    # 5. Эвакуация математики лосса из-под автокаста строго во float32 с изоляцией градиентов
    target_flow = pack_latents_to_patches((noise - latents).detach()).float().to(device=device)
    pred_tensor_f32 = pred_tensor.float()
#--------- Окончание блока №3_PATCH

    # Расчет маски весов и стабильный Huber/MSE шаг
    weight_mask = (1.0 / (1.0 - t_attr.float().view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(device=device)
    loss_active = (F.mse_loss(pred_tensor_f32, target_flow, reduction="none") * weight_mask).mean()
    
    # Дифференциальный спуск Autograd-графа
    (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()
    
#---------- Старт блока №6_TELEMETRY_FIX
    # 6. Настоящая раздельная телеметрия осей без строковых заглушек альтеров
    with torch.no_grad():
        m_pred = pred_tensor_f32. mean(). item()
        std_pred = pred_tensor_f32. std(). item()
        vram_active = torch. cuda. memory_allocated() / ( 1024 ** 3)
        vram_reserved = torch. cuda. memory_reserved() / ( 1024 ** 3)
        
        print(f"\n📊 [Кадр {frame_idx}] t_noise={t_attr.item():.3f} | Loss={loss_active.item():.4f}")
        print(f"📡 [ПРИБОРЫ] Физическая форма кадра в VRAM: {list(target_flow.shape)}")
        print(f"📡 [ПРИБОРЫ] Геометрия выхода ядра: {list(pred_tensor_f32.shape)}")
        print(f"📊 M_pred={m_pred:.4f} ±{std_pred:.4f} | VRAM Акт/Резерв: {vram_active:.2f}/{vram_reserved:.2f} GB\n")
#--------- Окончание блока №6_TELEMETRY_FIX
        
    # 7. Извлечение безопасных CPU-клонов до тотального выжигания памяти кадра
    loss_val = loss_active.detach().clone()
    t_attr_cpu = t_attr.detach().cpu()
    pred_tensor_cpu = pred_tensor_f32.detach().cpu()
    target_flow_cpu = target_flow.detach().cpu()
    
    # 8. Тотальная вакуумная зачистка тяжелых тензоров кадра из памяти
    del pred_tensor, pred_tensor_f32, weight_mask, loss_active, target_flow
    del latents, prompt_embeds, t_raw, t_attr, noise, noisy_latents, packed_noisy_latents
    del img_ids, txt_ids_aligned, kwargs_mask
    
    return loss_val, t_attr_cpu, pred_tensor_cpu, target_flow_cpu
# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №2 ===
