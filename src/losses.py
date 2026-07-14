import torch
import torch.nn as nn

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, x_1, condition, text_encoder=None):
        """
        ычисление потери для Pixel-space Flow Matching (Rectified Flow).
        x_1: еальные изображения из датасета (мангал), тензор [B, 3, H, W]
        condition: Текстовые описания (капшены с триггером)
        """
        batch_size = x_1.size(0)
        device = x_1.device
        dtype = x_1.dtype

        # 1. енерируем чистый гауссов шум x_0 такого же размера, как картинка
        x_0 = torch.randn_like(x_1)

        # 2. Сэмплируем случайный временной шаг t от 0 до 1 для каждого элемента в батче
        t = torch.rand(batch_size, device=device, dtype=dtype)
        
        # азворачиваем t до размеров [B, 1, 1, 1], чтобы перемножить с тензорами картинок
        t_blended = t.view(batch_size, 1, 1, 1)

        # 3. инейная интерполяция траектории между шумом и оригиналом
        x_t = (1.0 - t_blended) * x_0 + t_blended * x_1

        # 4. елевой вектор скорости (Target Velocity)
        target_velocity = x_1 - x_0

        # 5. мбеддинг текста (ондиционирование)
        text_embedding = condition 
        if text_encoder is not None:
            text_embedding = text_encoder(condition)

        # 6. редсказание модели
        pred_velocity = model(x_t, t, text_embedding)

        # 7. Считаем среднеквадратичную ошибку (MSE Loss)
        loss = torch.mean((pred_velocity - target_velocity) ** 2)

        return loss
