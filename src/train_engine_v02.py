# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №1: СИ-ЩИТ И ВАКУУМ ЭНКОДЕРОВ ===
# === ИНЖЕКТ КОРНЯ И ПРЕВЕНТИВНЫЙ СИ-ЩИТ ===
import os
import sys
# Силой заталкиваем корень проекта в пути поиска Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Намертво глушим ROCm до импорта тяжелых либ
os.environ.update({"QUANTO_DISABLE_CPP_EXT": "1", "HF_DISABLE_COMPILING": "1", "FORCE_CUDA": "1", "USE_ROCM": "0"})
for var in ["ROCM_HOME", "HIP_PATH", "ROCM_PATH"]: os.environ.pop(var, None)

import gc
import torch
import diffusers.utils.import_utils as du
from torch.optim import AdamW
from config import TrainConfig
from lora_core_v02 import FluxLoraCoreV02
from get_dataloader_v02 import get_dataloader_v02

# --- СИ-ЩИТ: АННИГИЛЯЦИЯ ROCM/HIP ---
os.environ.update({"QUANTO_DISABLE_CPP_EXT": "1", "HF_DISABLE_COMPILING": "1", "FORCE_CUDA": "1", "USE_ROCM": "0"})
for var in ["ROCM_HOME", "HIP_PATH", "OLLAMA_LLM_LIBRARY", "ROCM_PATH"]: os.environ.pop(var, None)
torch.version.hip, torch.version.rocm = None, None
du.is_rocm_available = lambda: False
du.is_torch_rocm_available = lambda: False

# Оптимизатор
try:
    from ao_optim_monolith_v02 import AdamW8bit
    optim_class, use_8bit = AdamW8bit, True
except:
    optim_class, use_8bit = AdamW, False

def init_components():
    """Инициализация ядра Chroma и VRAM-менеджмент."""
    print("[Т] Инициализация: train_engine_v02")
    
    # 1. Загрузка данных и модели
    dataloader = get_dataloader_v02()
    model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # [МАНЕВР]: Выгрузка T5XXL (освобождаем ~8 ГБ VRAM)
    if hasattr(model, "text_encoder"):
        print("\n***** UNLOADING T5XXL *****")
        del model.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim_class(trainable_params, lr=TrainConfig.LEARNING_RATE)

    # Contiguous
    for p in trainable_params:
        if p.ndim == 2: p.data = p.data.contiguous()
            
    print("[УСПЕХ] T5 выгружен, оптимизатор готов.")
    return dataloader, model, optimizer
# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №1 ===

# === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2А: ЦИКЛЫ И ВРЕМЕННОЙ ИНЖЕКТОР ===
def run_main_loop(dataloader, model, optimizer):
    """Маршевый контур плавки мантиссы Хромы с защитой от WDDM оверсвапа."""
    import time
    import sys
    import os
    import torch
    import gc
    from config import TrainConfig
    from step_executor_v02 import execute_single_frame_step
    from telemetry_logger import FluxTelemetryTracker

    # 1. Инициализация косинусного планировщика и глобального тахометра шагов
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS),
        eta_min=1e-6
    )
    
    global_step = 0
    last_log_time = time.time()
    telemetry = FluxTelemetryTracker()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 2. Маршевые вложенные циклы плавки
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, mega_batch in enumerate(dataloader):
            total_frames = mega_batch["latents"].shape[0]
            
            for frame_idx in range(total_frames):
                global_step += 1
                
                # [ВРЕМЕННОЙ ИНЖЕКТОР]: Нативный покадровый шаг через Узел 16x16 патчей
                loss_val, t_attr_cpu, pred_tensor_cpu, target_flow_cpu = execute_single_frame_step(
                    mega_batch=mega_batch,
                    frame_idx=frame_idx,
                    device=torch.device("cuda"),
                    lora_model=model
                )
                
                # [ВЫРАВНИВАНИЕ ЗРЕНИЯ]: Запись чистых метрик из CPU-доменов в логгер
                telemetry.accumulate_step(
                    t_attr=t_attr_cpu,
                    pred_tensor=pred_tensor_cpu,
                    target_tensor=target_flow_cpu,
                    current_loss=loss_val.item()
                )
                
                # Передаем управление в Блок 2Б (Стабилизация градиентов, оптимизация и клининг)

# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №2А ===

                # === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2Б-1: ОПТИМИЗАЦИЯ И ТЕЛЕМЕТРИЯ ===
                # 3. Такт оптимизации и жесткий клиппинг градиентов
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # Проверка градиентов на конечность
                if not all(torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Фиксация весов LoRA и очистка VRAM
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()

                # 4. Телеметрия (упрощенная версия)
                with torch.no_grad():
                    current_loss = loss_val.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                    allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
                    
                    # Логирование основных метрик
                    print(f"📊 Шаг: {global_step} | Loss: {current_loss:.4f} | VRAM: {allocated_vram:.2f} GB")
                    print("─" * 40)
                # Передаем управление в Блок 2Б-2
                # === КОНЕЦ СЛУЖЕБНОГО БЛОКА №2Б-1 ===
                # === НАЧАЛО СЛУЖЕБНОГО БЛОКА №2Б-2: ФИНАЛИЗАЦИЯ ДВИЖКА ===
                # 5. Сброс агрегированных логов и чекпоинтинг
                if global_step % TrainConfig.SAVE_STEPS == 0:
                    print(f"💾 [Т] Чекпоинт на шаге {global_step}...")
                    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
                    torch.save(lora_state, os.path.join(TrainConfig.OUTPUT_DIR, f"lora_{global_step}.safetensors"))

                    # Валидация (Врубаем eval режим)
                    model.eval()
                    with torch.no_grad(), torch.inference_mode():
                        from generate_v02 import run_inference_v02
                        run_inference_v02(model, global_step)
                    model.train()

        # Шаг планировщика вынесен за цикл батчей эпохи
        scheduler.step()
        print(f"✅ Эпоха {epoch} успешно завершена.")
    print("🔱 ПЛАВКА МАНТИССЫ РЕАКТОРА ЗАВЕРШЕНА.")

if __name__ == "__main__":
    dataloader, model, optimizer = init_components()
    run_main_loop(dataloader, model, optimizer)
# === КОНЕЦ СЛУЖЕБНОГО БЛОКА №2Б-2 И ВСЕГО МОДУЛЯ ===
