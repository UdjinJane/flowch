# === НАЧАЛО МОНОЛИТА: src/get_dataloader_v02.py ===
import torch
from config import TrainConfig

class FakeChromaDataset:
    """Генератор бесконечных тактических батчей мантиссы Хромы."""
    def __init__(self):
        # Настраиваем геометрию тензоров под каноничные 512x512 и T5XXL
        self.resolution = getattr(TrainConfig, "RESOLUTION", 512)
        self.max_txt_len = getattr(TrainConfig, "MAX_SEQUENCE_LENGTH", 128)
        self.batch_size = getattr(TrainConfig, "BATCH_SIZE", 1)
        
        # Размерность латентов Chroma: 16 каналов, нарезка 1/8 от разрешения
        self.latent_dim = self.resolution // 8

    def __iter__(self):
        return self

    def __next__(self):
        # Генерируем фейковые тренировочные тензоры для удержания плавки
        latents = torch.randn(
            self.batch_size, 16, self.latent_dim, self.latent_dim, 
            dtype=torch.bfloat16
        )
        prompt_embeds = torch.randn(
            self.batch_size, self.max_txt_len, 4096, 
            dtype=torch.bfloat16
        )
        
        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds
        }

def get_dataloader_v02():
    """Точка сопряжения: возвращает маршевый поток данных для train_engine."""
    print("📦 [РЕАКТОР] Автономный тактический даталоадер успешно развернут")
    return FakeChromaDataset()
# === КОНЕЦ МОНОЛИТА ===
