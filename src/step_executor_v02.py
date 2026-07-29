import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import TrainConfig
from model_runner_v02 import run_lora_model_step

def pack_latents_to_patches(latents):
    """Каноническая упаковка латентов Хромы под 16x16 патчи Метрополии."""
    b, c, h, w = latents.shape
    # Пересобираем оси: patch_size=16 (вместо ошибочного 2x2 у альтеров)
    p = 16
    return latents.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)

def generate_chroma_img_ids(height, width, device):
    """Генерация трехмерных RoPE идентификаторов кадра под длину Хромы."""
    p = 16
    h_patches, w_patches = height // p, width // p
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device, dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    return img_ids.view(1, -1, 3)

# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2: МАРШЕВЫЙ ШАГ И СМАРТ-ТЕЛЕМЕТРИЯ ===
def execute_single_frame_step(mega_batch, frame_idx, device, lora_model):
    """Полный герметичный отсек маршевого шага кадра Хромы с 7-портовой стыковкой."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from config import TrainConfig
    from model_runner_v02 import run_lora_model_step

    # 1. Извлечение и подготовка тензоров кадра
    latents = mega_batch["latents"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
    prompt_embeds = mega_batch["prompt_embeds"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
    
    # 2. Математика Rectified Flow (динамический сдвиг мантиссы шума, shift=3.0)
    t_raw = torch.rand(1, device=device, dtype=torch.bfloat16)
    t_attr = (3.0 * t_raw) / (1.0 + 2.0 * t_raw)
    noise = torch.randn_like(latents)
    t_model_scale = (t_attr.clone() * 1000.0).to(device=device, dtype=torch.bfloat16)
    
    noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
    packed_noisy_latents = pack_latents_to_patches(noisy_latents)
    
    # 3. Нарезка позиционных сеток RoPE под нативный 7-портовый узел Метрополии
    _, _, h_l, w_l = latents.shape
    img_ids = generate_chroma_img_ids(h_l, w_l, device)
    txt_ids_aligned = torch.zeros(1, prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
    
    kwargs_mask = {"txt_mask": torch.ones((1, prompt_embeds.shape[1]), device=device, dtype=torch.bfloat16)}
    
#---------- Старт блока №2_UPDATED
    # 4. Прямой проход с градиентным чекпоинтингом через 7 портов (БЕЗ pooled_projections)
    with torch. amp. autocast( device_type="cuda", dtype= torch. bfloat16):
        pred_tensor = checkpoint(
            run_lora_model_step,
            lora_model,
            kwargs_mask,
            packed_noisy_latents,
            t_model_scale,
            prompt_embeds,
            txt_ids_aligned,
            img_ids,
            use_reentrant= False
        )
#--------- Окончание блока №2_UPDATED

        
    # 5. Эвакуация математики лосса из-под автокаста строго во float32
    target_flow = pack_latents_to_patches(noise - latents).float().to(device=device)
    pred_tensor_f32 = pred_tensor.float()
    
    # Расчет маски весов и стабильный Huber/MSE шаг
    weight_mask = (1.0 / (1.0 - t_attr.float().view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(device=device)
    loss_active = (F.mse_loss(pred_tensor_f32, target_flow, reduction="none") * weight_mask).mean()
    
    # Дифференциальный спуск Autograd-графа
    (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()
    
    # 6. Космофлотская средневзвешенная смарт-телеметрия (моменты прогноза и полки VRAM)
    with torch.no_grad():
        m_pred = pred_tensor_f32.mean().item()
        std_pred = pred_tensor_f32.std().item()
        vram_active = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        print(f"📊 [Кадр {frame_idx}] t_noise={t_attr.item():.3f} | Loss={loss_active.item():.4f} | "
              f"M_pred={m_pred:.4f}±{std_pred:.4f} | VRAM Акт/Резерв: {vram_active:.2f}/{vram_reserved:.2f} GB")
        
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
