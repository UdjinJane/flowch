import os
import sys
import gc
import time
import json
import shutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import TrainConfig
from generate_v02 import run_inference_v02
from dataset_v02 import get_dataloader_v02
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

def pack_latents_to_patches(latents):
    # latents shape: [B, C, H, W] -> Flux требует [B, (H/2)*(W/2), C*4]
    b, c, h, w = latents.shape
    assert h % 2 == 0 and w % 2 == 0, f"Разрешение должно быть кратно 2, получили {h}x{w}"
    
    # Перестраиваем тензор в патчи 2x2
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2)
    return latents

def generate_flux_img_ids(batch_size, height, width, device):
    # Генерируем координатную сетку для патчей 2x2 изображения
    h_patches = height // 2
    w_patches = width // 2
    
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    
    img_ids = img_ids.view(-1, 3).repeat(batch_size, 1, 1)
    return img_ids

def main_train_loop():
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    
    # 1. Принудительная зачистка старого мусора
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    
    # 2. Инициализация датасета
    print("[Т] Запуск загрузчика кэшированных эмбеддингов...")
    dataloader = get_dataloader_v02()
    
    # 3. Инициализация LoRA контура
    print("[Т] Прогрев и инжекция LoRA адаптеров...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # Жесткая фиксация параметров: учим строго LoRA слои
    trainable_params = []
    for name, param in lora_model.named_parameters():
        if "lora_" in name and any(t in name for t in TrainConfig.TARGET_MODULES):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    print(f"[УСПЕХ] Зафиксировано обучаемых тензоров: {len(trainable_params)}")
    
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    
    # Настройки логирования
    log_file_path = os.path.join(TrainConfig.OUTPUT_DIR, "train_logs.txt")
    os.makedirs(TrainConfig.OUTPUT_DIR, exist_ok=True)
    
    global_step = 0
    
    # 4. Главная плавка (Цикл по эпохам)
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        lora_model.train()
        
        # Сброс генератора случайных чисел для стабильности эпохи
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, batch in enumerate(dataloader):
            global_step += 1
            
            # Извлекаем кэшированные латенты и эмбеддинги
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            pooled_projections = batch["pooled_projections"].to(device=device, dtype=torch.bfloat16)
            
            # Формируем шум и таймстепы
            noise = torch.randn_like(latents)
            # Задаем случайный шаг по траектории Rectified Flow [0.0, 1.0]
            t_attr = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16)
            
            # Прямое смешивание по закону Rectified Flow ODE
            # x_t = (1 - t) * x_0 + t * noise
            broadcast_t = t_attr.view(-1, 1, 1, 1)
            noisy_latents = (1.0 - broadcast_t) * latents + broadcast_t * noise
            
            # Целевой вектор скорости (flow target): v = noise - x_0
            target_flow = noise - latents
            
            # Переводим латенты и таргет в паттерн Flux-пачей (3D тензоры)
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Генерируем пространственные маркеры (разрешение берем из конфига - 512px)
            img_ids = generate_flux_img_ids(latents.shape[0], TrainConfig.RESOLUTION, TrainConfig.RESOLUTION, device)
            
            # Текстовые ID заполняем нулями, так как у нас фиксированные кэшированные эмбеддинги
            txt_ids = torch.zeros(latents.shape[0], prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
            
            # Передаем управление в runner шаг
            pred_tensor = run_lora_model_step(
                lora_model=lora_model,
                batch=batch,
                packed_noisy_latents=packed_noisy_latents,
                timesteps_attr=t_attr,
                prompt_embeds=prompt_embeds,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                img_ids=img_ids
            )
            
            # Принудительная синхронизация типов перед математикой MSE
            pred_tensor = pred_tensor.to(dtype=torch.bfloat16)
            packed_target_flow = packed_target_flow.to(dtype=torch.bfloat16)
            
            # Контроль геометрии: [B, 1024, 64] против [B, 1024, 64]
            if pred_tensor.shape != packed_target_flow.shape:
                # Если ранер вернул объединенный контекст (с текстом), режем строго под картинку
                pred_tensor = pred_tensor[:, :packed_target_flow.shape[1], :]
            
            # Расчет среднеквадратичной ошибки
            loss = F.mse_loss(pred_tensor, packed_target_flow, reduction="mean")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[КРИТ] Обнаружен взрыв градиентов или NaN на шаге {global_step}! Плавку останавливаем.")
                sys.exit(1)
                
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            
            # ЗАПУСК ОБРАТНОГО НАПРАВЛЕНИЯ ПЛАЗМЫ (ГРАДИЕНТЫ ПОШЛИ!)
            loss.backward()
            
            # Накопление градиентов и маршевый шаг оптимизатора
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                # Жесткая обрезка по регламенту V02_STABLE_PLUS против FP8 Underflow / Overflow
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            # Зачистка локального VRAM кэша на каждом шаге под лимит 21.0 GB
            torch.cuda.empty_cache()
            
            # Периодический репорт в консоль и бортовой журнал
            if global_step % 10 == 0:
                current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                log_msg = f"[ОТК] Шаг: {global_step} | Эпоха: {epoch} | MSE Лосс: {current_loss:.4f}\n"
                print(log_msg.strip())
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write(log_msg)
                    
            # 5. Рубеж инференса и сохранения контрольных точек
            if global_step % TrainConfig.SAVE_STEPS == 0:
                print(f"[Т] Достигнут рубеж сохранения. Запекаем чекпоинт на шаге {global_step}...")
                checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                
                # Сохраняем строго веса LoRA адаптеров
                lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                torch.save(lora_state_dict, checkpoint_path)
                print(f"[УСПЕХ] Контрольный пирог сохранен: {checkpoint_path}")
                
                # Тестовая плавка одного изображения для визуального контроля
                print("[Т] Запуск тестовой генерации кадра на орбите...")
                lora_model.eval()
                with torch.no_grad():
                    run_inference_v02(global_step)
                lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку всех эпох. LoRA готова к вылету!")

if __name__ == "__main__":
    main_train_loop()
