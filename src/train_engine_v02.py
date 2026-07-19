import os
import sys
import gc
import shutil
import torch
import torch.nn.functional as F
# ... (остальные кастомные импорты v02)
from torch.optim import AdamW
from config import TrainConfig
from generate_v02 import run_inference_v02
from dataset_v02 import get_dataloader_v02
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

def pack_latents_to_patches(latents):
# ... (весь остальной код функций без изменений)

    """
    Упаковывает 4D-латенты [B, C, H, W] в 3D-тензор Flux [B, L, C*4].
    """
    b, c, h, w = latents.shape
    # Перестраиваем тензор в пространственные патчи 2x2
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2)
    return latents

def generate_flux_img_ids(height, width, device):
    """
    Генерирует 2D-тензор координат [num_patches, 3] для RoPE.
    """
    h_p, w_p = height // 2, width // 2
    img_ids = torch.zeros(h_p, w_p, 3, device=device)
    img_ids[..., 1] = torch.arange(h_p, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_p, device=device)[None, :]
    return img_ids.view(-1, 3)

def main_train_loop():
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    
    # 1. Принудительная зачистка и инициализация девайса (строго до вызовов модулей!)
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda") 
    
    # 2. Инициализация датасета и модели
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # 3. Фиксация LoRA-параметров (только lora_ слои)
    trainable_params = []
    for name, param in lora_model.named_parameters():
        if "lora_" in name and any(t in name for t in TrainConfig.TARGET_MODULES):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    print(f"[УСПЕХ] Зафиксировано обучаемых тензоров: {len(trainable_params)}")
    
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    
    # Логирование
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    global_step = 0
    # 4. Главная плавка (Цикл по эпохам и батчам)
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, batch in enumerate(dataloader):
            global_step += 1
            
            # Извлекаем данные, Rectified Flow (шум+t), 2D-RoPE (img/txt_ids)
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            b_size = latents.shape[0]
            
            noise = torch.randn_like(latents)
            t_attr = torch.rand(b_size, device=device, dtype=torch.bfloat16)
            
            # Батчинг noisy_latents и генерация 2D-координат
            packed_noisy_latents = pack_latents_to_patches((1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise)
            img_ids = generate_flux_img_ids(TrainConfig.RESOLUTION, TrainConfig.RESOLUTION, device)
            txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
            
            # Маршевый вызов lora раннера
            pred_tensor = run_lora_model_step(
                lora_model=lora_model, batch=batch, packed_noisy_latents=packed_noisy_latents,
                timesteps_attr=t_attr, prompt_embeds=prompt_embeds,
                pooled_projections=torch.zeros(b_size, 768, device=device, dtype=torch.bfloat16),
                txt_ids=txt_ids, img_ids=img_ids
            )
            # Принудительная стабилизация типов и расчет MSE [1.6]
            pred_tensor = pred_tensor.to(dtype=torch.bfloat16)
            target_flow = (noise - latents).to(dtype=torch.bfloat16)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Контроль геометрии и расчет лосса
            if pred_tensor.shape != packed_target_flow.shape:
                pred_tensor = pred_tensor[:, :packed_target_flow.shape[1], :]
            loss = F.mse_loss(pred_tensor, packed_target_flow, reduction="mean")
            
            # Обработка взрыва градиентов
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[КРИТ] Взрыв градиентов на шаге {global_step}!")
                sys.exit(1)
                
            # Обратное распространение и оптимизация [1.6]
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            # Очистка VRAM и логирование [1.6]
            torch.cuda.empty_cache()
            if global_step % 10 == 0:
                current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                print(f"[ОТК] Шаг: {global_step} | MSE: {current_loss:.4f}")
                with open(os.path.join(TrainConfig.LOGS_DIR, "train_logs.txt"), "a") as lf:
                    lf.write(f"Шаг: {global_step} | Loss: {current_loss:.4f}\n")
                    
            # 5. Сохранение и инференс [1.6]
            if global_step % TrainConfig.SAVE_STEPS == 0:
                checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"lora_step_{global_step}.safetensors")
                lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                torch.save(lora_state_dict, checkpoint_path)
                
                # Тестовая генерация
                lora_model.eval()
                with torch.no_grad():
                    run_inference_v02(global_step)
                lora_model.train()

    print("[УСПЕХ] Обучение завершено!")

if __name__ == "__main__":
    main_train_loop()
