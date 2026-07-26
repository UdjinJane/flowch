import os
import sys

# --- СИ-ЩИТ КЭПА: ТОТАЛЬНАЯ АННИГИЛЯЦИЯ ROCM/HIP ---
os.environ["QUANTO_DISABLE_CPP_EXT"] = "1"
os.environ["HF_DISABLE_COMPILING"] = "1"
os.environ["FORCE_CUDA"] = "1"
os.environ["USE_ROCM"] = "0"
for var in ["ROCM_HOME", "HIP_PATH", "OLLAMA_LLM_LIBRARY", "ROCM_PATH"]:
    os.environ.pop(var, None)

import torch
torch.version.hip = None
torch.version.rocm = None

# Принудительное отключение ROCM в diffusers
import diffusers.utils.import_utils as du
du.is_rocm_available = lambda: False
du.is_torch_rocm_available = lambda: False

# --- ОСНОВНЫЕ ИМПОРТЫ ---
import gc
import torch.nn.functional as F
from torch.optim import AdamW

from config import TrainConfig
from lora_core_v02 import FluxLoraCoreV02

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
    # Инициализация и очистка памяти перед стартом
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    import shutil
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    
    # Загрузка данных и LoRA
    from dataset_v02 import get_dataloader_v02
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # Определение обучаемых параметров
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    
    # Настройка оптимизатора (AdamW или AdamW8bit)
    if USING_8BIT_OPTIM:
        optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)

    # Активация градиентного чекпоинтинга для экономии VRAM
    if hasattr(lora_model, "gradient_checkpointing_enable"):
        lora_model.gradient_checkpointing_enable()
    elif hasattr(lora_model, "base_model") and hasattr(lora_model.base_model.model, "gradient_checkpointing_enable"):
        lora_model.base_model.model.gradient_checkpointing_enable()
    print("[УСПЕХ] Градиентный чекпоинтинг маршевого двигателя успешно взведен!")

    # Настройка планировщика
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS),
        eta_min=1e-6
    )

    # Инициализация телеметрии с локальным подтягиванием модуля
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(TrainConfig.LOGS_DIR, "train_logs.txt")
    
    from telemetry_logger import FluxTelemetryTracker
    telemetry = FluxTelemetryTracker()


    global_step = 0
    
    # Локальная инжекция системного таймера для изоляции NameError
    import time
    last_log_time = time.time()

    # Вход в основной цикл
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        epoch_start_time = time.time()
        
        for step, mega_batch in enumerate(dataloader):
            all_latents = mega_batch["latents"]
            all_embeds = mega_batch["prompt_embeds"]
            total_frames = all_latents.shape[0]
            
            # Нарезка мега-батча на отдельные кадры
            for frame_idx in range(total_frames):
                global_step += 1
                latents = all_latents[frame_idx:frame_idx+1].to(device=device, dtype=torch.bfloat16)
                prompt_embeds = all_embeds[frame_idx:frame_idx+1].to(device=device, dtype=torch.bfloat16)
                # Логит-нормальный замер времени и масштабирование под [0-1000]
                t_attr = torch.sigmoid(torch.randn(1, device=device) * 1.0).to(dtype=torch.bfloat16)
                noise = torch.randn_like(latents)
                t_model_scale = (t_attr * 1000.0).to(device=device, dtype=torch.bfloat16)

                # Формирование зашумленных данных и сетки патчей
                noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
                packed_noisy_latents = pack_latents_to_patches(noisy_latents)

                # Прецизионное нарезание позиционных img_ids под длину токенов кадра Хромы
                num_latent_tokens = packed_noisy_latents.shape[1]
                img_ids = torch.zeros(1, num_latent_tokens, 3, device=device, dtype=torch.bfloat16)
                
                # Выравнивание текстовых позиционных индексов
                txt_len = prompt_embeds.shape[1]
                txt_ids_aligned = torch.zeros(1, txt_len, 3, device=device, dtype=torch.bfloat16)
                
                # Аварийная очистка кэша перед входом в тяжелое ядро для защиты от оверсвапа
                torch.cuda.empty_cache()
                # Инициализация замера времени прямого прохода
                torch.cuda.synchronize()
                t_fwd_start = time.time()

                # Выполнение шага модели (PEFT + Rectified Flow)
                pred_tensor = run_lora_model_step(
                    lora_model,
                    {"txt_mask": torch.ones((1, txt_len), device=device, dtype=torch.bfloat16)},
                    packed_noisy_latents,
                    t_model_scale,
                    prompt_embeds,
                    torch.zeros(1, 768, device=device, dtype=torch.bfloat16),
                    txt_ids_aligned,
                    img_ids
                )

                # Завершение замера и расчет целевого потока
                torch.cuda.synchronize()
                print(f"[КОНТРОЛЬ] Время прямого прохода: {time.time() - t_fwd_start:.4f} сек.")
                target_flow = pack_latents_to_patches(latents - noise).to(dtype=torch.bfloat16, device=device)

                # Решейпинг вывода (256 -> 64 канала) и применение весов
                pred_tensor_64 = pred_tensor.view(-1, pred_tensor.shape[1], 64, 4).mean(dim=-1) if pred_tensor.shape[-1] == 256 else pred_tensor
                weight_mask = (1.0 / (1.0 - t_attr.view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(dtype=torch.float32, device=device)
                # Прецизионный расчет лосса и передача метрик в самописец
                loss_active = (F.mse_loss(pred_tensor_64.float(), target_flow.float(), reduction="none") * weight_mask).mean()
                loss = loss_active.detach().clone().to(torch.bfloat16)
                telemetry.accumulate_step(t_attr, pred_tensor_64, target_flow, loss)

                # Выполнение обратного прохода с учетом шага накопления градиентов
                (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()

                # Такт оптимизации и жесткий клиппинг аномальных градиентов
                if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=TrainConfig.MAX_NORM)
                    
                    # Проверка градиентов на конечность (NaN/Inf предохранитель)
                    for param in trainable_params:
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            print("[КРИТ] Обнаружен взрыв градиентов! Аварийная остановка.")
                            sys.exit(1)
                            
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Полный сброс фрагментированного кэша Windows для подавления Shared RAM
                    torch.cuda.empty_cache()

                # Расширенный рапорт по приборам на каждом шаге
                current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                allocated_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)
                
                elapsed_time = time.time() - last_log_time
                speed = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
                last_log_time = time.time()

                console_msg = (
                    f"[ОТК] Шаг: {global_step} | Эпоха: {epoch} | "
                    f"MSE Лосс: {current_loss:.4f} | Скорость: {speed:.2f} it/s | "
                    f"VRAM Active: {allocated_vram:.2f} GB | Reserved: {reserved_vram:.2f} GB"
                )
                print(console_msg)

                # Сброс логов на накопитель космошхуны
                telemetry.flush_aggregated_log(global_step, epoch)
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write(f"Шаг: {global_step} | Loss: {current_loss:.4f} | Speed: {speed:.2f} it/s\n")

                # --- РУБЕЖ ЧЕКПОИНТИНГА И ИЗОЛИРОВАННОЙ ВАЛИДАЦИИ ---
                if global_step % TrainConfig.SAVE_STEPS == 0:
                    print(f"[Т] Рубеж фиксации. Запекаем чекпоинт на шаге {global_step}...")
                    checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                    lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                    torch.save(lora_state_dict, checkpoint_path)

                    # Намертво запечатываем инференс от утечек Autograd графа в Shared VRAM
                    lora_model.eval()
                    with torch.no_grad():
                        with torch.inference_mode():
                            run_inference_v02(
                                loaded_transformer=lora_model,
                                current_step=global_step
                            )
                    torch.cuda.empty_cache()
                    lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку всех эпох. Контур чист!")

if __name__ == "__main__":
    main_train_loop()
