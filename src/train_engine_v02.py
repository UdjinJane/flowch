import os
import sys
import gc
import time
import shutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import TrainConfig
from generate_v02 import run_inference_v02
from dataset_v02 import get_dataloader_v02
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

# --- АВТОМАТИЗИРОВАННАЯ ЗАЩИТА ОТ КРИВЫХ РУК (ИНЖЕКЦИЯ ЗОЛОТА V02) ---
try:
    from ao_optim_monolith_v02 import AdamW8bit
    USING_8BIT_OPTIM = True
except Exception as e:
    print(f"[ВНИМАНИЕ] Ошибка инжекции 8-bit монолита: {e}")
    print("[ОТК] Аварийный протокол: переключаюсь на стандартный float32 AdamW.")
    USING_8BIT_OPTIM = False

def pack_latents_to_patches(latents):
    b, c, h, w = latents.shape
    assert h % 2 == 0 and w % 2 == 0, f"Разрешение должно быть кратно 2, получили {h}x{w}"
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    return latents.flatten(1, 2)

def generate_flux_img_ids(height, width, device):
    h_patches, w_patches = height // 2, width // 2
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    return img_ids.view(-1, 3)

def main_train_loop():
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    
    print("[Т] Запуск загрузчика кэшированных эмбеддингов...")
    dataloader = get_dataloader_v02()
    
    print("[Т] Прогрев и инжекция LoRA адаптеров...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
        # Принудительная активация чекпоинтинга для полной разгрузки VRAM
    if hasattr(lora_model, "enable_gradient_checkpointing"):
        lora_model.enable_gradient_checkpointing()
    elif hasattr(lora_model, "get_base_model") and hasattr(lora_model.get_base_model(), "enable_gradient_checkpointing"):
        lora_model.get_base_model().enable_gradient_checkpointing()

    trainable_params = []
    for name, param in lora_model.named_parameters():
        if "lora_" in name and any(t in name for t in TrainConfig.TARGET_MODULES):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    print(f"[УСПЕХ] Зафиксировано обучаемых тензоров: {len(trainable_params)}")
    
    # Автоматический селектор оптимизатора на основе защиты контура
    if USING_8BIT_OPTIM:
        print("[УСПЕХ] Реактор успешно переведен на экономное int8-топливо (AdamW8bit V02).")
        optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(TrainConfig.LOGS_DIR, "train_logs.txt")
    global_step = 0
    last_log_time = time.time()  # <--- Фиксируем базовую метку времени здесь!
    
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        #------------------------------------------------------------------------------------------------------------
        # Фиксируем время старта эпохи
        epoch_start_time = time.time()
        
        for step, mega_batch in enumerate(dataloader):
            # Извлекаем тензоры из даталоадера (весь датасет)
            all_latents = mega_batch["latents"]
            all_embeds = mega_batch["prompt_embeds"]
            total_frames = all_latents.shape[0]

            # Нарезаем мега-батч на отдельные кадры (BATCH_SIZE = 1)
            for frame_idx in range(total_frames):
                global_step += 1
            
                # Вырезаем по 1 кадру (с размерностью B=1)
                latents = all_latents[frame_idx:frame_idx+1].to(device=device, dtype=torch.bfloat16)
                prompt_embeds = all_embeds[frame_idx:frame_idx+1].to(device=device, dtype=torch.bfloat16)
            
                # ... (Подготовка тензоров: шум, таймстепы)
                noise = torch.randn_like(latents)
                t_attr = torch.rand(1, device=device, dtype=torch.bfloat16)
                # ... (Упаковка и генерация ID)
                packed_noisy_latents = pack_latents_to_patches((1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise)
                img_ids = generate_flux_img_ids(latents.shape[2], latents.shape[3], device).to(torch.bfloat16)
                
                # изолированный мини-батч текущего кадра          
                current_batch = {"latents": latents, "prompt_embeds": prompt_embeds}

                        
                #
                # --- СНАЙПЕРСКИЙ ВЫЗОВ РАННЕРА V02 (СТРОКИ 94-98) ---
                txt_ids = torch.zeros((prompt_embeds.shape[1], 3), device=device, dtype=torch.bfloat16)
                pred_tensor = run_lora_model_step(
                    lora_model=lora_model,
                    batch=current_batch,
                    packed_noisy_latents=packed_noisy_latents, # ВОССТАНОВЛЕНО
                    timesteps_attr=t_attr,                     # ВОССТАНОВЛЕНО
                    prompt_embeds=prompt_embeds,
                    pooled_projections=torch.zeros(1, 768, device=device, dtype=torch.bfloat16),
                    txt_ids=txt_ids,
                    img_ids=img_ids
                )
                
                # ----------------------------------------------------

            
            pred_tensor = pred_tensor.to(dtype=torch.bfloat16)
            target_flow = (noise - latents).to(dtype=torch.bfloat16)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # --- ИСПРАВЛЕННЫЙ СНАЙПЕРСКИЙ СРЕЗ КАНАЛОВ (СТРОКА 126) ---
            # Принудительно выравниваем и длину последовательности (dim 1), и каналы (dim 2)
            if pred_tensor.shape != packed_target_flow.shape:
                pred_tensor = pred_tensor[:, :packed_target_flow.shape[1], :packed_target_flow.shape[2]]
            # ----------------------------------------------------------
            
            loss = F.mse_loss(pred_tensor, packed_target_flow, reduction="mean")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[КРИТ] Обнаружен взрыв градиентов на шаге {global_step}!")
                sys.exit(1)
                
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            # Принудительно вычищаем системные ссылки Python и сбрасываем кэш CUDA на каждом шаге
            gc.collect()
            torch.cuda.empty_cache()

            #
            if global_step % 10 == 0:
                current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                allocated_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)
                
                # Точный скользящий замер за 10 шагов
                elapsed_time = time.time() - last_log_time
                speed = 10.0 / elapsed_time if elapsed_time > 0 else 0.0
                last_log_time = time.time() # Сброс строго в точке замера


        
                # Формируем расширенный рапорт для Мистральчика
                console_msg = (
                    f"[ОТК] Шаг: {global_step} | Эпоха: {epoch} | "
                    f"MSE Лосс: {current_loss:.4f} | Скорость: {speed:.2f} it/s | "
                    f"VRAM Active: {allocated_vram:.2f} GB | Reserved: {reserved_vram:.2f} GB"
                )
                file_msg = f"Шаг: {global_step} | Loss: {current_loss:.4f} | Speed: {speed:.2f}it/s | VRAM: {allocated_vram:.2f}GB\n"
        
                print(console_msg)  # В консоль летит красивый рапорт
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write(file_msg)  # В файл пишется чистая строка без дублей
            
            # --- РУБЕЖ СОХРАНЕНИЯ И ГЕНЕРАЦИИ СЭМПЛОВ ---
            # --- ИСПРАВЛЕННЫЙ РУБЕЖ СОХРАНЕНИЯ (СТРОГО ПО ШАГАМ) ---
            if global_step % TrainConfig.SAVE_STEPS == 0:

                print(f"[Т] Рубеж фиксации. Запекаем чекпоинт на шаге {global_step}...")
                checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                torch.save(lora_state_dict, checkpoint_path)
                
                # Врубаем тестовую генерацию кадра для Кэпа
                lora_model.eval()
                with torch.no_grad():
                    run_inference_v02(
                        loaded_transformer=lora_model, 
                        current_step=global_step, 
                        text_embedding=prompt_embeds
                    )
                lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку всех эпох. Контур чист!")

if __name__ == "__main__":
    main_train_loop()
