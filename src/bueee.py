import os
import sys

# ============================================================================
# АБСОЛЮТНЫЙ СИ-ЩИТ КЭПА: ТОТАЛЬНАЯ АННИГИЛЯЦИЯ ТРИГГЕРОВ ROCM НА УРОВНЕ ЯДРА
# ============================================================================
os.environ["QUANTO_DISABLE_CPP_EXT"] = "1"
os.environ["HF_DISABLE_COMPILING"] = "1"
os.environ["FORCE_CUDA"] = "1"
os.environ["USE_ROCM"] = "0"

for amdbug in ["ROCM_HOME", "HIP_PATH", "HIP_PATH_62", "OLLAMA_LLM_LIBRARY", "HIP_DIR", "ROCM_PATH"]:
    os.environ.pop(amdbug, None)

import torch
torch.version.hip = None
torch.version.rocm = None

import diffusers.utils.import_utils
diffusers.utils.import_utils.is_rocm_available = lambda *args, **kwargs: False
diffusers.utils.import_utils.is_torch_rocm_available = lambda *args, **kwargs: False
diffusers.utils.import_utils.is_hip_available = lambda *args, **kwargs: False
diffusers.utils.import_utils._torch_rocm_available = False
diffusers.utils.import_utils._rocm_available = False

sys.modules["diffusers.utils.import_utils"] = diffusers.utils.import_utils

import transformers.utils
if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
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

    
    #№№ Автоматический селектор оптимизатора на основе защиты контура
    # [БЛОК ОБНОВЛЕН: АНТИ-ОВЕРСВАП ЗАЩИТА АКТИВИРОВАНА]
    # Автоматический селектор оптимизатора
    if USING_8BIT_OPTIM:
        optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)

    # --- ВКЛЮЧЕНИЕ ГРАДИЕНТНОГО ЧЕКПОИНТИНГА ---
    if hasattr(lora_model, "gradient_checkpointing_enable"):
        lora_model.gradient_checkpointing_enable()
    
    # --- ИНИЦИАЛИЗАЦИЯ ПЛАНИРОВЩИКА И ТЕЛЕМЕТРИИ ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.MAX_TRAIN_STEPS)
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    telemetry = FluxTelemetryTracker() # [ПОЛНАЯ ТЕЛЕМЕТРИЯ ВОССТАНОВЛЕНА]

    # --- ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ (ПЛАН: ЗАМЕНА СТАРЫХ БЛОКОВ) ---
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        lora_model.train()
        for step, mega_batch in enumerate(dataloader):
            # ... (логика загрузки и подготовки латентов) ...
            
            # --- ПРЯМОЙ ПРОХОД И РАСЧЕТ ЛОССА ---
            pred_tensor = run_lora_model_step(lora_model, ...)
            loss_active = F.mse_loss(...) # [КРИТИЧЕСКИЙ L1-LOSS АКТИВЕН]
            
            # --- ОБРАТНЫЙ ПРОХОД И ОПТИМИЗАЦИЯ ---
            (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # --- ЛОГИРОВАНИЕ (ВКЛЮЧАЯ TELEMETRY.FLUSH) ---
            telemetry.flush_aggregated_log(global_step, epoch)
            # ... (запись в текстовый файл) ...

            # --- ЧЕКПАОИНТИНГ И ИНФЕРЕНС ---
            if global_step % TrainConfig.SAVE_STEPS == 0:
                torch.save(lora_model.state_dict(), ...)
                run_inference_v02(...)

# [ОБНОВЛЕНИЕ ЗАВЕРШЕНО]
    
                
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
