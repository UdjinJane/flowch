# === НАЧАЛО МОДУЛЯ: src/step_executor_v02.py (БЛОК 1) ===
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import TrainConfig

def pack_latents_to_patches(latents):
    """Упаковка латентов, оптимизированная под 2x2 патчи."""
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)


def generate_flux_img_ids(height, width, device):
    """Генерация идентификаторов изображения для корректной позиционной кодировки."""
    h_patches, w_patches = height // 2, width // 2
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    return img_ids.view(-1, 3)
# === ФИНАЛ МОДУЛЯ: src/step_executor_v02.py (БЛОК 1) ===
# === НАЧАЛО МОДУЛЯ: src/step_executor_v02.py (БЛОК 2) ===
def execute_single_frame_step(mega_batch, frame_idx, device, lora_model):
    """Герметичный инженерный отсек одного кадра. Полная изоляция Autograd-цепей."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from config import TrainConfig
    from model_runner_v02 import run_lora_model_step

    # Перенос тензоров кадра на устройство
    latents = mega_batch["latents"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
    prompt_embeds = mega_batch["prompt_embeds"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)

    # Реализация динамического сдвига мантиссы шума (shift=3.0) по чертежам Лденов
    t_raw = torch.rand(1, device=device, dtype=torch.bfloat16)
    t_attr = (3.0 * t_raw) / (1.0 + 2.0 * t_raw)
    
    noise = torch.randn_like(latents)
    t_model_scale = (t_attr.clone() * 1000.0).to(device=device, dtype=torch.bfloat16)
    noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
    packed_noisy_latents = pack_latents_to_patches(noisy_latents)

    # Подготовка сеток и заглушки pooled_projections (bfloat16) согласно 8-портовой схеме
    _, _, h_l, w_l = latents.shape
    h_p, w_p = h_l // 2, w_l // 2
    grid_ids = torch.zeros(h_p, w_p, 3, device=device, dtype=torch.bfloat16)
    grid_ids[..., 1] = torch.arange(h_p, device=device)[:, None]
    grid_ids[..., 2] = torch.arange(w_p, device=device)[None, :]
    img_ids = grid_ids.view(1, -1, 3)
    
    txt_ids_aligned = torch.zeros(1, prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
    kwargs_mask = {"txt_mask": torch.ones((1, prompt_embeds.shape[1]), device=device, dtype=torch.bfloat16)}
    pooled_projections_fake = torch.zeros(1, 768, device=device, dtype=torch.bfloat16)

    # Прямой проход с autocast bfloat16
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_tensor = checkpoint(
            run_lora_model_step, 
            lora_model, 
            kwargs_mask, 
            packed_noisy_latents, 
            t_model_scale, 
            prompt_embeds, 
            pooled_projections_fake, 
            txt_ids_aligned, 
            img_ids, 
            use_reentrant=False
        )

    # Вынос математики лосса из-под автокаста строго во float32
    target_flow = pack_latents_to_patches(noise - latents).float().to(device=device)
    pred_tensor_f32 = pred_tensor.float()
    
    if pred_tensor_f32.shape[-1] == 256:
        pred_tensor_64 = pred_tensor_f32.view(-1, pred_tensor_f32.shape[1], 64, 4).mean(dim=-1)
    else:
        pred_tensor_64 = pred_tensor_f32
        
    weight_mask = (1.0 / (1.0 - t_attr.float().view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(device=device)
    loss_active = (F.mse_loss(pred_tensor_64, target_flow, reduction="none") * weight_mask).mean()

    # Backward в стабильном float32
    (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()

    # Извлечение безопасных CPU-клонов до выжигания тензоров из VRAM
    loss_val = loss_active.detach().clone()
    t_attr_cpu = t_attr.detach().cpu()
    pred_tensor_64_cpu = pred_tensor_64.detach().cpu()
    target_flow_cpu = target_flow.detach().cpu()

    # Тотальное вакуумное удаление тяжелых тензоров кадра
    del pred_tensor, pred_tensor_f32, pred_tensor_64, weight_mask, loss_active, target_flow
    del latents, prompt_embeds, t_raw, t_attr, noise, noisy_latents, packed_noisy_latents
    del grid_ids, img_ids, txt_ids_aligned, kwargs_mask, pooled_projections_fake

    return loss_val, t_attr_cpu, pred_tensor_64_cpu, target_flow_cpu

# === КОНЕЦ МОДУЛЯ: src/step_executor_v02.py ===
