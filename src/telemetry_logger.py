import os
from datetime import datetime
import torch
from config import TrainConfig

class FluxTelemetryTracker:
    """Автономный модуль скользящей агрегации метрик для Flux-реактора V02_STABLE."""

    import os
from datetime import datetime
import torch
from config import TrainConfig

class FluxTelemetryTracker:
    """Модуль агрегации метрик, использующий TrainConfig.LOGS_DIR."""

    def __init__(self):
        # Используем путь из конфига, хардкод устранён
        self.target_dir = TrainConfig.LOGS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(self.target_dir, f"session_{timestamp}.txt")
        os.makedirs(self.target_dir, exist_ok=True)
        self.pred_buffer = []
        self.target_buffer = []
        self.loss_buffer = []

    def accumulate_step(self, t_attr, pred_tensor, target_tensor, current_loss):
        """Сбор метрик без воровства VRAM."""
        with torch.no_grad():
            self.pred_buffer.append(pred_tensor.mean().item())
            self.target_buffer.append(target_tensor.mean().item())
            self.loss_buffer.append(current_loss.item() if hasattr(current_loss, "item") else float(current_loss))

    def flush_aggregated_log(self, global_step, epoch):
        """Сброс усреднённых метрик в файл."""
        if not self.pred_buffer: return
        n = len(self.pred_buffer)
        log_line = f"[СЕССИЯ] Шаг: {global_step} | Loss: {sum(self.loss_buffer)/n:.6f}"
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        self.pred_buffer.clear(); self.target_buffer.clear(); self.loss_buffer.clear()


        # Полная очистка трюмов для следующего батча агрегации
        self.pred_buffer.clear()
        self.target_buffer.clear()
        self.loss_buffer.clear()
        self.t_attr_buffer.clear()
