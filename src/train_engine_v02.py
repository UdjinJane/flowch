import os
import sys
from types import ModuleType

# ============================================================================
# УЛЬТИМАТИВНЫЙ ТОТАЛЬНЫЙ СУПЕР-ХАК КЭПА: ПРИНУДИТЕЛЬНОЕ ОПЕРЕЖАЮЩЕЕ ОСЛЕПЛЕНИЕ
# ============================================================================
os.environ["QUANTO_DISABLE_CPP_EXT"] = "1"
os.environ["HF_DISABLE_COMPILING"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["FORCE_CUDA"] = "1"
os.environ["USE_ROCM"] = "0"

for amdbug in ["ROCM_HOME", "HIP_PATH", "HIP_PATH_62", "OLLAMA_LLM_LIBRARY", "HIP_DIR", "ROCM_PATH"]:
    os.environ[amdbug] = ""
    if amdbug in os.environ:
        del os.environ[amdbug]

# Принудительно импортируем оригинальные утилиты, сохраняя все константы (ENV_VARS_TRUE_VALUES и др.)
import diffusers.utils.import_utils

# Лазерно переписываем только методы проверки ROCm/HIP на жесткое False
diffusers.utils.import_utils.is_rocm_available = lambda *args, **kwargs: False
diffusers.utils.import_utils.is_torch_rocm_available = lambda *args, **kwargs: False
diffusers.utils.import_utils.is_hip_available = lambda *args, **kwargs: False

# Намертво фиксируем подмененный живой модуль в системном кэше Python
sys.modules["diffusers.utils.import_utils"] = diffusers.utils.import_utils
# ============================================================================
# Дальше идут оригинальные импорты управляющего дирижёра
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
from telemetry_logger import FluxTelemetryTracker



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

    # Оптимизатор забирает параметры, чьи флаги requires_grad уже монолитно выставлены внутри lora_core
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    print(f"[УСПЕХ] Зафиксировано обучаемых тензоров адаптера: {len(trainable_params)}")

    
    # Автоматический селектор оптимизатора на основе защиты контура
    if USING_8BIT_OPTIM:
        print("[УСПЕХ] Реактор успешно переведен на экономное int8-топливо (AdamW8bit V02).")
        optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)

    # --- ИНИЦИАЛИЗАЦИЯ КОСИНУСНОГО ПЛАНИРОВЩИКА (ПЛАТИНОВАЯ КНИГА БЛОК 5) ---
    # Тушит LR до 1e-6 к финалу TOTAL_STEPS, пробивая бетонное плато лосса
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS),
        eta_min=1e-6
    )
    print(f"[УСПЕХ] Планировщик CosineAnnealingLR взведен на T_max={scheduler.T_max}")

    
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(TrainConfig.LOGS_DIR, "train_logs.txt")
    telemetry = FluxTelemetryTracker()
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
            
                # Логит-нормальный замер времени и масштабирование под [0-1000]
                # === ИНЖЕКЦИЯ CHROMA V05: ЧАСТЬ 1 (Подготовка и вызов) ===
                t_attr = torch.sigmoid(torch.randn(1, device=device) * 1.0).to(dtype=torch.bfloat16)
                noise = torch.randn_like(latents)
                t_model_scale = (t_attr * 1000.0).to(device=device, dtype=torch.bfloat16)

                # Формирование зашумленных данных и сетки
                noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
                packed_noisy_latents = pack_latents_to_patches(noisy_latents)
                img_ids = generate_flux_img_ids(latents.shape[2], latents.shape[3], device).to(torch.bfloat16)


                # === СНАЙПЕРСКИЙ ПОЗИЦИОННЫЙ ВЫЗОВ РАННЕРА V05 С ТАЙМЕРОМ ФАЗЫ ===
                # --- Вскрываем обстановку! ----
                # === ПРЕДПУСКОВОЙ РАДАР СТРУКТУРЫ ШАГА (АНТИ-ОВЕРСВАП АУДИТ) ===
                if global_step == 1:
                    print("\n🚨🚨🚨 [РАДАР] ВНИМАНИЕ! ПЕРЕХВАТ ТЕНЗОРОВ ПЕРЕД ВХОДОМ В ЯДРО: 🚨🚨🚨")
                    print(f" -> [ВХОД] latents shape: {latents.shape} | dtype: {latents.dtype} | Device: {latents.device} | Эквивалент VRAM: {latents.element_size() * latents.nelement() / (1024**2):.4f} MB")
                    print(f" -> [ВХОД] prompt_embeds shape: {prompt_embeds.shape} | dtype: {prompt_embeds.dtype} | Device: {prompt_embeds.device} | Эквивалент VRAM: {prompt_embeds.element_size() * prompt_embeds.nelement() / (1024**2):.4f} MB")
                    print(f" -> [ВХОД] packed_noisy_latents shape: {packed_noisy_latents.shape} | dtype: {packed_noisy_latents.dtype} | Device: {packed_noisy_latents.device} | Эквивалент VRAM: {packed_noisy_latents.element_size() * packed_noisy_latents.nelement() / (1024**2):.4f} MB")
                    print(f" -> [ВХОД] img_ids shape: {img_ids.shape} | dtype: {img_ids.dtype} | Device: {img_ids.device}")
                    print(f" -> [VRAM БАЗА] Выделено CUDA перед forward: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
                    print("=========================================================================\n")

                torch.cuda.synchronize()
                t_fwd_start = time.time()
                
                pred_tensor = run_lora_model_step(
                    lora_model,
                    {"txt_mask": torch.ones((1, prompt_embeds.shape[1]), device=device, dtype=torch.bfloat16)},
                    packed_noisy_latents,
                    t_model_scale,
                    prompt_embeds,
                    torch.zeros(1, 768, device=device, dtype=torch.bfloat16),
                    torch.zeros((prompt_embeds.shape[1], 3), device=device, dtype=torch.bfloat16),

                    img_ids
                )
                torch.cuda.synchronize()
                t_fwd_end = time.time()

                # Фикс рассинхрона: выводим время чистого fwd-прохода на абсолютно каждом шагу обучения
                print(f"[КОНТРОЛЬ] Время чистого прямого прохода ядра: {t_fwd_end - t_fwd_start:.4f} сек.")

                # === КОНЕЦ ЧАСТИ 1 ===
                
                # === КАНbackgroundИЧЕСКОЕ ВЫРАВНИВАНИЕ МАНТИССЫ RECTIFIED FLOW V11 ===
                # 1. Расчет истинного направления потока (честные 64 канала упакованных пикселей)
                raw_target_flow = pack_latents_to_patches(latents - noise).to(dtype=torch.bfloat16, device=device)
                target_flow = raw_target_flow
                
                # 2. Безопасное сжатие выхлопа раннера из 256 внутренних каналов трансформера обратно в 64 канала кадра
                # Мы прессуем тензор [1, 4096, 256] -> [1, 4096, 64, 4] и берем среднее по четверкам признаков
                if pred_tensor.shape[-1] == 256:
                    # Берем реальную длину последовательности Хромы (shape[1]) прямо из тензора рантайма
                    pred_tensor_64 = pred_tensor.view(-1, pred_tensor.shape[1], 64, 4).mean(dim=-1)
                else:

                    pred_tensor_64 = pred_tensor
                    
                # 3. Активация защиты от Аномалии Песка (динамический вес по сигме)
                weight_mask = (1.0 / (1.0 - t_attr.view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(dtype=torch.float32, device=device)
                
                # 4. Прецизионный расчет лосса строго на канонических 64 каналах Flux
                loss_active = (F.mse_loss(pred_tensor_64.float(), target_flow.float(), reduction="none") * weight_mask).mean()
                loss = loss_active.detach().clone().to(torch.bfloat16)

              
                # ----------- ТЕЛЕМЕТРИЯ_НАШЕ ВСЁ V12 -----------------------------
                # Передаем строго выровненную 64-канальную мантиссу, чтобы у логгера не сорвало клапаны
                telemetry.accumulate_step(t_attr, pred_tensor_64, target_flow, loss)


                # 5. Обратная волна градиентов по каноническому шагу накопления
                (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()

                # === КОНЕЦ ИНЖЕКЦИИ CHROMA V05 ===

                if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # ПРЕДОХРАНИТЕЛЬ ГРАДИЕНТОВ (STRICT VALIDATION)
                for param in trainable_params:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"[КРИТ] Обнаружен взрыв или затухание градиентов (NaN/Inf) перед шагом оптимизатора!")
                        sys.exit(1)
                
                optimizer.step()
                optimizer.zero_grad()

                # ---- END ПРЕДОХРАНИТЕЛЬ ГРАДИЕНТОВ (STRICT VALIDATION)
            #
            # ТАКТ ПЛАНИРОВЩИКА: Безопасное увядание плазмы шага строго ПОСЛЕ step() [1.10]
            scheduler.step()

            # ВЫПРЯМЛЕННЫЙ ВЫВОД ТЕЛЕМЕТРИИ ДЛЯ КЭПА НА КАЖДОМ ШАГУ
            current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
            allocated_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved_vram = torch.cuda.memory_reserved(device) / (1024 ** 3)
            
            # Замер скорости шага (теперь работает без задержек на каждой итерации)
            elapsed_time = time.time() - last_log_time
            speed = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            last_log_time = time.time() # Сброс счетчика тахометра

            # Формируем расширенный рапорт по приборам
            console_msg = (
                f"[ОТК] Шаг: {global_step} | Эпоха: {epoch} | "
                f"MSE Лосс: {current_loss:.4f} | Скорость: {speed:.2f} it/s | "
                f"VRAM Active: {allocated_vram:.2f} GB | Reserved: {reserved_vram:.2f} GB"
            )
            file_msg = f"Шаг: {global_step} | Loss: {current_loss:.4f} | Speed: {speed:.2f} it/s | VRAM: {allocated_vram:.2f} GB\n"
            print(console_msg)

            # Выдергиваем глубокую статистику памяти CUDA на первом шаге
            if global_step == 1:
                print("\n==================================================")
                print("[ТЕЛЕМЕТРИЯ CUDA] Спектральный анализ аллокатора:")
                print(f" -> Max Allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB")
                print(f" -> Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")
                print("==================================================\n")

            telemetry.flush_aggregated_log(global_step, epoch)
            with open(log_file_path, "a", encoding="utf-8") as lf:
                lf.write(file_msg)
                
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
                    # === СТЫКОВКА ИНФЕРЕНСА CHROMA V07 ===
                    # Передаем только трансформер и шаг. Подмену VAE на FakeVAE сделаем внутри generate_v02!
                    run_inference_v02(
                        loaded_transformer=lora_model,
                        current_step=global_step
                    )

                # === КОНЕЦ СТЫКОВКИ ===

                lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку всех эпох. Контур чист!")

if __name__ == "__main__":
    main_train_loop()
