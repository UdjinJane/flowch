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
    from torch.utils.checkpoint import checkpoint
    
    # --- АКТИВАЦИЯ СТЕРИЛЬНОГО РЕЖИМА КУЗНЕЦОВ (src/train_engine_v02.py) ---
    # Принудительно отключаем внутреннее накопление Autograd для замороженных FP8 SVD весов
    lora_model.base_model.model.requires_grad_(False) if hasattr(lora_model, "base_model") else lora_model.requires_grad_(False)
    
    # Направляем градиентный поток СТРОГО на обучаемые параметры LoRA
    for param in trainable_params:
        param.requires_grad_(True)
        if param.ndim == 2:
            param.data = param.data.contiguous()  # Исключаем фрагментацию памяти Windows при backward
            
    print("[УСПЕХ] Магистрали Autograd очищены от холостых FP8-активаций. Линия герметична!")

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
            
            # Локальная инжекция шлюза исполнения для подавления NameError
            from model_runner_v02 import run_lora_model_step
            
            # === МАРШЕВЫЙ ДВИГАТЕЛЬ V02: ФРАГМЕНТ 1 ИЗ 2 (SUMM) ===
            for frame_idx in range(total_frames):
                global_step += 1
                # Очистка градиентов и кэша
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
                # Подготовка данных (bf16)
                latents = all_latents[frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
                prompt_embeds = all_embeds[frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
                
                # Rectified Flow & Latent Packing
                t_attr = torch.sigmoid(torch.randn(1, device=device) * 1.0).to(torch.bfloat16)
                noise = torch.randn_like(latents)
                noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
                packed_noisy_latents = pack_latents_to_patches(noisy_latents)
                
                # Генерация ID (img/txt)
                h_p, w_p = latents.shape[2] // 2, latents.shape[3] // 2
                grid_ids = torch.zeros(h_p, w_p, 3, device=device, dtype=torch.bfloat16)
                grid_ids[..., 1] = torch.arange(h_p, device=device)[:, None]
                grid_ids[..., 2] = torch.arange(w_p, device=device)[None, :]
                img_ids = grid_ids.view(1, -1, 3)
                
                txt_ids_aligned = torch.zeros(1, prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
                
                # Синхронизация и подготовка Kwargs
                torch.cuda.synchronize()
                kwargs_mask = {"txt_mask": torch.ones((1, prompt_embeds.shape[1]), device=device, dtype=torch.bfloat16)}
                pooled_projections_fake = torch.zeros(1, 768, device=device, dtype=torch.bfloat16)
                # === ФИНАЛ: ФРАГМЕНТ 1 ===
                
                # === МАРШЕВЫЙ ДВИГАТЕЛЬ V02 СТАРТ: ФРАГМЕНТ 2 ИЗ 2 ===
                # Автоматическая смешанная точность (bfloat16) для оптимизации VRAM
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Forward pass с checkpointing для экономии памяти
                    pred_tensor = checkpoint(
                        run_lora_model_step,
                        lora_model, kwargs_mask, packed_noisy_latents, t_model_scale,
                        prompt_embeds, pooled_projections_fake, txt_ids_aligned, img_ids,
                        use_reentrant=False
                    )

                torch.cuda.synchronize()
                target_flow = pack_latents_to_patches(latents - noise).to(dtype=torch.bfloat16, device=device)

                # Вычисление взвешенного MSE loss и backward pass
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_tensor_64 = pred_tensor.view(-1, pred_tensor.shape[1], 64, 4).mean(dim=-1) if pred_tensor.shape[-1] == 256 else pred_tensor
                    weight_mask = (1.0 / (1.0 - t_attr.view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(dtype=torch.bfloat16, device=device)
                    loss_active = (F.mse_loss(pred_tensor_64, target_flow, reduction="none") * weight_mask).mean()
                    loss_active.backward()

                # Обновление параметров и очистка
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=TrainConfig.MAX_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                    

                # --- ТЕЛЕМЕТРИЯ И ЧЕКПОИНТИНГ (ФИНАЛ) ---
                # [Логирование метрик и сохранение LoRA модели]
                if global_step % TrainConfig.SAVE_STEPS == 0:
                    print(f"[Т] Чекпоинт {global_step}...")
                    checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                    torch.save({k: v for k, v in lora_model.state_dict().items() if "lora_" in k}, checkpoint_path)
                    
                    # [Временное переключение на eval для генерации]
                    lora_model.eval()
                    with torch.no_grad(), torch.inference_mode():
                        from generate_v02 import run_inference_v02
                        run_inference_v02(loaded_transformer=lora_model, current_step=global_step)
                    lora_model.train()

        scheduler.step()
    print("[УСПЕХ] Реактор завершил плавку. Контур чист!")

if __name__ == "__main__":
    main_train_loop()
# === БЛОК ДАННЫХ V02 ФИНАЛ: КОНЕЦ ФАЙЛА ===
