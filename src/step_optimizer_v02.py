# === НАЧАЛО СЛУЖЕБНОГО МОДУЛЯ: src/step_optimizer_v02.py ===
import gc
import torch
from config import TrainConfig

def run_optimizer_and_telemetry(model, optimizer, loss_val, global_step, epoch, frame_idx):
    """Изолированный тактовый узел оптимизации параметров LoRA и зачистки кэша CUDA."""
    
    # 1. Жесткий скалярный клиппинг градиентов параметров LoRA (канон 1.0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    
    # 2. Аварийный предохранитель Метрополии от взрыва градиентов (NaN/Inf)
    if not all(torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None):
        print(f"🚨 [РЕАКТОР] Обнаружены субнормальные числа на шаге {global_step}! Пропуск кадра.")
        optimizer.zero_grad(set_to_none=True)
        return False

    # 3. Фиксация весов LoRA и немедленный покадровый клининг VRAM под Windows
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()

    # 4. МГНОВЕННЫЙ КОСМОФЛОТСКИЙ РАПОРТ НА МОСТИК
    with torch.no_grad():
        m_loss = loss_val.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
        allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_vram = torch.cuda.memory_reserved() / (1024 ** 3)
        
        print(f"📊 [МАРШ] Шаг: {global_step} | Эпоха: {epoch} | Кадр: {frame_idx}")
        print(f" └── Loss кадра: {m_loss:.4f} | VRAM Активная: {allocated_vram:.2f} GB")
        print(f" └── Кэш CUDA: {reserved_vram:.2f} GB | Статус: КОНТУР СТАБИЛЕН")
        print("─" * 60)
        
    return True
# === КОНЕЦ СЛУЖЕБНОГО МОДУЛЯ ===
