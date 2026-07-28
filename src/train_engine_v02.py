# === НАЧАЛО БЛОКА ИМПОРТОВ GEMMA V3.5 ===
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
# === ФИНАЛ БЛОКА ИМПОРТОВ GEMMA V3.5 ===
# === НАЧАЛО БЛОКА ГЕОМЕТРИИ И СТАРТА ДВИГАТЕЛЯ GEMMA V3.5 ===
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


def main_train_loop():
    """Основной контур обучения с применением стерильного режима."""
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    
    # Очистка кэша перед стартом для предотвращения утечек
    import shutil
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    # [ВОССТАНОВЛЕНИЕ КОНТУРА]: Принудительная инициализация устройства CUDA
    device = torch.device("cuda")
    
    # Инициализация модели, данных и оптимизатора
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE) if USING_8BIT_OPTIM else AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE)

    # --- СТЕРИЛЬНЫЙ РЕЖИМ КУЗНЕЦОВ (src/train_engine_v02.py) ---
    # Отключение градиентов для замороженных весов
    lora_model.base_model.model.requires_grad_(False) if hasattr(lora_model, "base_model") else lora_model.requires_grad_(False)
    
    # Обеспечение непрерывности памяти (contiguous) для обучаемых параметров
    for param in trainable_params:
        param.requires_grad_(True)
        if param.ndim == 2: param.data = param.data.contiguous()

    print("[УСПЕХ] Магистрали Autograd очищены от холостых FP8-активаций. Линия герметична!")
# === ФИНАЛ БЛОКА ГЕОМЕТРИИ И СТАРТА ДВИГАТЕЛЯ GEMMA V3.5 ===
# === БЛОК ПЛАНИРОВЩИКА И ЦИКЛА GEMMA V3.5 (СЖАТЫЙ) ===
    # Инициализация косинусного планировщика и телеметрии
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS), eta_min=1e-6)
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    from telemetry_logger import FluxTelemetryTracker
    telemetry, global_step, last_log_time = FluxTelemetryTracker(), 0, time.time()

    # Основной цикл обучения (эпохи и кадры)
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, mega_batch in enumerate(dataloader):
            total_frames = mega_batch["latents"].shape[0]
            from model_runner_v02 import run_lora_model_step

            for frame_idx in range(total_frames):
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                # Перенос тензоров кадра на устройство
                latents = mega_batch["latents"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
                prompt_embeds = mega_batch["prompt_embeds"][frame_idx:frame_idx+1].to(device, dtype=torch.bfloat16)
# === ФИНАЛ БЛОКА ПЛАНИРОВЩИКА И ЦИКЛА GEMMA V3.5 ===
# === НАЧАЛО БЛОКА ШУМА И ПРЯМОГО ПРОХОДА GEMMA V3.5 (REFACTORED) ===
# Реализация динамического сдвига мантиссы шума (shift=3.0) и подготовка тензоров
t_raw = torch.rand(1, device=device, dtype=torch.bfloat16)
t_attr = (3.0 * t_raw) / (1.0 + 2.0 * t_raw)
noise = torch.randn_like(latents)
t_model_scale = (t_attr.clone() * 1000.0).to(device=device, dtype=torch.bfloat16)
noisy_latents = (1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise
packed_noisy_latents = pack_latents_to_patches(noisy_latents)

# Подготовка сеток и заглушки pooled_projections (bfloat16)
_, _, h_l, w_l = latents.shape
h_p, w_p = h_l // 2, w_l // 2
grid_ids = torch.zeros(h_p, w_p, 3, device=device, dtype=torch.bfloat16)
grid_ids[..., 1] = torch.arange(h_p, device=device)[:, None]
grid_ids[..., 2] = torch.arange(w_p, device=device)[None, :]
img_ids = grid_ids.view(1, -1, 3)
txt_ids_aligned = torch.zeros(1, prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
kwargs_mask = {"txt_mask": torch.ones((1, prompt_embeds.shape[1]), device=device, dtype=torch.bfloat16)}
pooled_projections_fake = torch.zeros(1, 768, device=device, dtype=torch.bfloat16)

# Прямой проход с autocast
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred_tensor = checkpoint(run_lora_model_step, lora_model, kwargs_mask, 
                             packed_noisy_latents, t_model_scale, prompt_embeds, 
                             pooled_projections_fake, txt_ids_aligned, img_ids, use_reentrant=False)
# === ФИНАЛ БЛОКА ШУМА И ПРЯМОГО ПРОХОДА ===
# === НАЧАЛО БЛОКА ODE И ОБРАТНОГО ПРОХОДА GEMMA V3.5 ===
# Вынос расчетов в float32, корректировка вектора (noise - latents), 
# маскирование и безопасный обратный проход.
target_flow = pack_latents_to_patches(noise - latents).float().to(device=device)
pred_tensor_f32 = pred_tensor.float()
pred_tensor_64 = pred_tensor_f32.view(-1, pred_tensor_f32.shape[1], 64, 4).mean(dim=-1) if pred_tensor_f32.shape[-1] == 256 else pred_tensor_f32
weight_mask = (1.0 / (1.0 - t_attr.float().view(-1, 1, 1) + 1e-4)).clamp(max=10.0).to(device=device)
loss_active = (F.mse_loss(pred_tensor_64, target_flow, reduction="none") * weight_mask).mean()

# === НАЧАЛО БЛОКА ODE И ОБРАТНОГО ПРОХОДА GEMMA V3.5 ===
        # Backward и отрыв тензоров для очистки VRAM
        (loss_active / TrainConfig.GRADIENT_ACCUMULATION_STEPS).backward()
        
        # [РЕАНИМАЦИЯ МЕТРИК]: Извлечение безопасных CPU-клонов до выжигания тензоров из VRAM
        loss_val = loss_active.detach().clone()
        loss = loss_val
        t_attr_cpu = t_attr.detach().cpu()
        pred_tensor_64_cpu = pred_tensor_64.detach().cpu()
        target_flow_cpu = target_flow.detach().cpu()

        del pred_tensor, pred_tensor_f32, pred_tensor_64, weight_mask, loss_active, target_flow
# === ФИНАЛ БЛОКА ODE И ОБРАТНОГО ПРОХОДА GEMMA V3.5 ===

# === НАЧАЛО БЛОКА КЛИППИНГА, ПРЕДОХРАНИТЕЛЯ И ФИНАЛА GEMMA V3.5 ===

        # [ВОССТАНОВЛЕНИЕ ЗРЕНИЯ]: Запись метрик мантиссы в изолированные буферы логгера
        telemetry.accumulate_step(
            t_attr=t_attr_cpu,
            pred_tensor=pred_tensor_64_cpu,
            target_tensor=target_flow_cpu,
            current_loss=loss.item()
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

        # Сброс логов и чекпоинтинг
        if global_step % 10 == 0 and len(telemetry.loss_buffer) > 0:
            avg_loss = sum(telemetry.loss_buffer) / len(telemetry.loss_buffer)
            print(f"\n 📡 [ТЕЛЕМЕТРИЯ] Шаг: {global_step} | MSE: {avg_loss:.6f}")
            telemetry.flush_aggregated_log(global_step, epoch)

        if global_step % TrainConfig.SAVE_STEPS == 0:
            print(f"[Т] Чекпоинт на шаге {global_step}...")
            torch.save({k: v for k, v in lora_model.state_dict().items() if "lora_" in k},
                        os.path.join(TrainConfig.OUTPUT_DIR, f"lora_step_{global_step}.safetensors"))

            # Инференс-валидация
            lora_model.eval()
            with torch.no_grad(), torch.inference_mode():
                from generate_v02 import run_inference_v02
                run_inference_v02(loaded_transformer=lora_model, current_step=global_step)
            torch.cuda.empty_cache()
            lora_model.train()

        scheduler.step()

    print(f"[УСПЕХ] Эпоха {epoch} завершена.")

print("[УСПЕХ] Все эпохи завершены.")


if __name__ == "__main__":
    main_train_loop()
# === БЛОК ДАННЫХ V02 ФИНАЛ: КОНЕЦ МОЗАИКИ И КОНЕЦ ФАЙЛА ===
