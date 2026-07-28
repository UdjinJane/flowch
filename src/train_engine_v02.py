# === НАЧАЛО МОДУЛЯ: src/train_engine_v02.py (БЛОК 1) ===
import os
import sys
import time

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

# --- ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ ---
import diffusers.utils.import_utils as du
du.is_rocm_available = lambda: False
du.is_torch_rocm_available = lambda: False
import gc
from torch.optim import AdamW
from config import TrainConfig
from lora_core_v02 import FluxLoraCoreV02
from dataset_v02 import get_dataloader_v02
from step_executor_v02 import execute_single_frame_step


try:
    from ao_optim_monolith_v02 import AdamW8bit
    USING_8BIT_OPTIM = True
except:
    USING_8BIT_OPTIM = False

def main_train_loop():
    """Основной контур диспетчера верхнего уровня."""
    print("[Т] Запуск легкого маршевого диспетчера: train_engine_v02")
    import shutil
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    
    # Стерильный режим кузнецов для LoRA
    optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE) if USING_8BIT_OPTIM else AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE)
    
    # Подготовка логгера
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    from telemetry_logger import FluxTelemetryTracker
    telemetry = FluxTelemetryTracker()
    
    # Фиксация градиентов
    for param in trainable_params:
        if param.ndim == 2: param.data = param.data.contiguous()
    print("[УСПЕХ] Диспетчер инициализирован. Переходим к циклам плавки.")
# === ФИНАЛ МОДУЛЯ: src/train_engine_v02.py (БЛОК 1) ===
# === НАЧАЛО МОДУЛЯ: src/train_engine_v02.py (БЛОК 2) ===
    # Инициализация косинусного планировщика и глобального тахометра шагов
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS), 
        eta_min=1e-6
    )
    global_step = 0
    last_log_time = time.time()

    # Маршевые вложенные циклы плавки мантиссы
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, mega_batch in enumerate(dataloader):
            total_frames = mega_batch["latents"].shape[0]

            for frame_idx in range(total_frames):
                global_step += 1

                # [ВРЕМЕННОЙ ИНЖЕКТОР]: Изолированный покадровый шаг через Узел 2
                loss_val, t_attr_cpu, pred_tensor_64_cpu, target_flow_cpu = execute_single_frame_step(
                    mega_batch=mega_batch,
                    frame_idx=frame_idx,
                    device=device,
                    lora_model=lora_model
                )

                # [ВОССТАНОВЛЕНИЕ ЗРЕНИЯ]: Запись чистых метрик из CPU-доменов в логгер
                telemetry.accumulate_step(
                    t_attr=t_attr_cpu,
                    pred_tensor=pred_tensor_64_cpu,
                    target_tensor=target_flow_cpu,
                    current_loss=loss_val.item()
                )

                # Такт оптимизации и жесткий клиппинг аномальных градиентов параметров LoRA
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Проверка градиентов на конечность (NaN/Inf предохранитель)
                for param in trainable_params:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print("[КРИТ] Обнаружен взрыв градиентов! Аварийная остановка.")
                        sys.exit(1)

                # Фиксация весов LoRA и немедленный сброс Autograd-накопления кадра
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # Расширенный рапорт по приборам на каждом шаге в консоль мостика
                current_loss = loss_val.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
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

                # Сброс логов и чекпоинтинг
                if global_step % 10 == 0 and len(telemetry.loss_buffer) > 0:
                    avg_loss = sum(telemetry.loss_buffer) / len(telemetry.loss_buffer)
                    print(f"\n 📡 [ТЕЛЕМЕТРИЯ] Шаг: {global_step} | MSE: {avg_loss:.6f}")
                    telemetry.flush_aggregated_log(global_step, epoch)

                if global_step % TrainConfig.SAVE_STEPS == 0:
                    print(f"[Т] Чекпоинт на шаге {global_step}...")
                    lora_state = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                    checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"lora_step_{global_step}.safetensors")
                    torch.save(lora_state, checkpoint_path)

                    # [ГЕРМЕТИЗАЦИЯ ВАЛИДАЦИИ]: Полная изоляция весов для защиты от WDDM оверсвапа
                    lora_model.eval()
                    with torch.no_grad(), torch.inference_mode():
                        # Принудительно очищаем граф перед импортом
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        from generate_v02 import run_inference_v02
                        run_inference_v02(loaded_transformer=lora_model, current_step=global_step)
                    
                    # Жесткое выжигание следов инференса из памяти перед возвратом в train
                    gc.collect()
                    torch.cuda.empty_cache()
                    lora_model.train()


                scheduler.step()

    print(f"[УСПЕХ] Эпоха {epoch} завершена.")

print("[УСПЕХ] Все эпохи завершены.")


if __name__ == "__main__":
    main_train_loop()
# === БЛОК ДАННЫХ V02 ФИНАЛ: КОНЕЦ МОЗАИКИ И КОНЕЦ ФАЙЛА ===
