import os
from datetime import datetime
import torch
from config import TrainConfig

class FluxTelemetryTracker:
    """Бортовой самописец расширенных метрик тензоров (Mean, Std, Min/Max) для Flux V02."""

    def __init__(self):
        target_dir = TrainConfig.LOGS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(target_dir, f"session_{timestamp}.txt")
        os.makedirs(target_dir, exist_ok=True)
        
        self.pred_buffer = []
        self.target_buffer = []
        self.loss_buffer = []
        self.t_attr_buffer = []
    def accumulate_step(self, t_attr, pred_tensor, target_tensor, current_loss):
        """Молча собирает прецизионные статистики тензоров, полностью изолируя Autograd."""
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
        """Рассчитывает скользящее среднее за 10 шагов и сбрасывает богатый лог на SSD."""
        if not self.pred_buffer:
            return

        n = len(self.pred_buffer)
        
        # Расчет усредненных метрик
        avg_metrics = {
            "pred_mean": sum(p["mean"] for p in self.pred_buffer) / n,
            "pred_std": sum(p["std"] for p in self.pred_buffer) / n,
            "target_mean": sum(t["mean"] for t in self.target_buffer) / n,
            "target_std": sum(t["std"] for t in self.target_buffer) / n,
            "loss": sum(self.loss_buffer) / n,
            "t_attr": sum(self.t_attr_buffer) / n,
            "min_p": min(p['min'] for p in self.pred_buffer),
            "max_p": max(p['max'] for p in self.pred_buffer),
            "min_t": min(t['min'] for t in self.target_buffer),
            "max_t": max(t['max'] for t in self.target_buffer)
        }

        # Формирование и запись богатой строки лога
        log_line = (
            f"[СЕССИЯ] Шаг: {global_step} | Эпоха: {epoch} | "
            f"Loss: {avg_metrics['loss']:.6f} | Время: {avg_metrics['t_attr']:.4f} | "
            f"Pred: ({avg_metrics['pred_mean']:.6f}±{avg_metrics['pred_std']:.6f}, [{avg_metrics['min_p']:.4f}, {avg_metrics['max_p']:.4f}]) | "
            f"Target: ({avg_metrics['target_mean']:.6f}±{avg_metrics['target_std']:.6f}, [{avg_metrics['min_t']:.4f}, {avg_metrics['max_t']:.4f}])"
        )

        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        # Очистка буферов
        self.pred_buffer.clear()
        self.target_buffer.clear()
        self.loss_buffer.clear()
        self.t_attr_buffer.clear()

