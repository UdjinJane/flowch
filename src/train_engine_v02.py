import os
import sys
import gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import TrainConfig
from generate_v02 import run_inference_v02
from dataset_v02 import get_dataloader_v02
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

def pack_latents_to_patches(latents):
    """
    Упаковывает 4D-латенты [B, C, H, W] в 3D-тензор Flux [B, L, C*4].
    Для разрешения 512px выдает форму [B, 1024, 64].
    """
    b, c, h, w = latents.shape
    assert h % 2 == 0 and w % 2 == 0, f"Разрешение должно быть кратно 2, получили {h}x{w}"
    
    # Перестраиваем тензор в пространственные патчи 2x2
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2)
    return latents

def generate_flux_img_ids(height, width, device):
    """
    Генерирует строго 2D-тензор координат [num_patches, 3] без батч-размерности.
    Устраняет критический сбой RoPE-эмбеддингов в diffusers.
    """
    h_patches = height // 2
    w_patches = width // 2
    
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    
    return img_ids.view(-1, 3)

def main_train_loop():
    # 1. Очистка кэша CUDA, загрузка данных, инициализация LoRA [1.6]
    gc.collect()
    torch.cuda.empty_cache()
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # 2. Фиксация весов: заморозка основного слоя, обучение только LoRA [1.2, 1.6]
    trainable_params = []
    for name, param in lora_model.named_parameters():
        if "lora_" in name and any(t in name for t in TrainConfig.TARGET_MODULES):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    # 3. Инициализация оптимизатора и логов [1.6]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    # 4. Главная плавка (Цикл по эпохам и батчам)
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        lora_model.train()
        for step, batch in enumerate(dataloader):
            # Кэширование данных
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            b_size = latents.shape[0]
            
            # Rectified Flow: шум и time
            noise = torch.randn_like(latents)
            t_attr = torch.rand(b_size, device=device, dtype=torch.bfloat16)
            noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
            
            # Упаковка в 3D-патчи
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            
            # 2D-RoPE: генерация img_ids и txt_ids
            img_ids = generate_flux_img_ids(TrainConfig.RESOLUTION, TrainConfig.RESOLUTION, device)
            
            # Вызов раннера
            pred_tensor = run_lora_model_step(
                lora_model, batch, packed_noisy_latents, t_attr, 
                prompt_embeds, img_ids=img_ids
            )

            # Принудительная стабилизация типов и расчет MSE Loss [1.6]
            pred_tensor = pred_tensor.to(dtype=torch.bfloat16)
            target_flow = (noise - latents).to(dtype=torch.bfloat16)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Срез тензора для совпадения размеров [1.6]
            if pred_tensor.shape != packed_target_flow.shape:
                pred_tensor = pred_tensor[:, :packed_target_flow.shape[1], :]
            
            loss = F.mse_loss(pred_tensor, packed_target_flow, reduction="mean")
            
            # Обработка NaN/Inf, обратное распространение и клиппинг [1.6]
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[КРИТ] Взрыв градиентов на {global_step}!")
                sys.exit(1)
                
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Накопление градиентов и шаг оптимизатора [1.6]
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            # Очистка кэша и логирование [1.6]
            torch.cuda.empty_cache()
            if global_step % 10 == 0:
                print(f"[ОТК] Шаг: {global_step} | Лосс: {loss.item():.4f}")
                    
            # 5. Сохранение чекпоинтов и визуальный инференс [1.6]
            if global_step % TrainConfig.SAVE_STEPS == 0:
                checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                torch.save(lora_state_dict, checkpoint_path)
                
                # Тестовая генерация [1.6]
                lora_model.eval()
                with torch.no_grad():
                    run_inference_v02(global_step)
                lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку. LoRA готова!")

if __name__ == "__main__":
    main_train_loop()
