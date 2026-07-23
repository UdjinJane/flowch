import os
from datetime import datetime
import torch

class FluxTelemetryTracker:
    """Автономный модуль скользящей агрегации метрик для Flux-реактора V02_STABLE."""

    def __init__(self, logs_dir="Z:\\flowch\\logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(logs_dir, f"session_{timestamp}.txt")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Стерильные буферы накопления метрик
        self.pred_buffer = []
        self.target_buffer = []
        self.loss_buffer = []
        self.t_attr_buffer = []

    def accumulate_step(self, t_attr, pred_tensor, target_tensor, current_loss):
        """Молча собирает метрики тензоров без воровства VRAM и CUDA-синхронизаций."""
        with torch.no_grad():
            pred_stats = {
                "mean": pred_tensor.mean().item(),
                "std": pred_tensor.std().item(),
                "min": pred_tensor.min().item(),
                "max": pred_tensor.max().item()
            }
            target_stats = {
                "mean": target_tensor.mean().item(),
                "std": target_tensor.std().item(),
                "min": target_tensor.min().item(),
                "max": target_tensor.max().item()
            }
            self.pred_buffer.append(pred_stats)
            self.target_buffer.append(target_stats)
            self.loss_buffer.append(current_loss.item() if hasattr(current_loss, "item") else float(current_loss))
            self.t_attr_buffer.append(t_attr.item() if hasattr(t_attr, "item") else float(t_attr))

    def flush_aggregated_log(self, global_step, epoch):
        """Проводит скользящую агрегацию за 10 шагов и сбрасывает строку в файл."""
        if not self.pred_buffer or not self.target_buffer:
            return

        n = len(self.pred_buffer)
        
        # Точное скользящее усреднение метрик
        pred_mean = sum(p["mean"] for p in self.pred_buffer) / n
        pred_std = sum(p["std"] for p in self.pred_buffer) / n
        
        target_mean = sum(t["mean"] for t in self.target_buffer) / n
        target_std = sum(t["std"] for t in self.target_buffer) / n
        
        loss_mean = sum(l for l in self.loss_buffer) / n
        t_attr_mean = sum(t for t in self.t_attr_buffer) / n

        log_line = (
            f"[СЕССИЯ] Шаг: {global_step} | Эпоха: {epoch} | "
            f"Loss: {loss_mean:.6f} | Время t: {t_attr_mean:.4f} | "
            f"Предсказания (Mean: {pred_mean:.6f}, Std: {pred_std:.6f}, Min/Max: {min(p['min'] for p in self.pred_buffer):.4f}/{max(p['max'] for p in self.pred_buffer):.4f}) | "
            f"Целевой поток (Mean: {target_mean:.6f}, Std: {target_std:.6f}, Min/Max: {min(t['min'] for t in self.target_buffer):.4f}/{max(t['max'] for t in self.target_buffer):.4f})"
        )

        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        # Полная очистка трюмов для следующего батча агрегации
        self.pred_buffer.clear()
        self.target_buffer.clear()
        self.loss_buffer.clear()
        self.t_attr_buffer.clear()
