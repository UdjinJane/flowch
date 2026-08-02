import os
import torch
from torch.utils.data import Dataset

class ChromaDataset(Dataset):
    """Сверхэкономный маршевый загрузчик латентов и текстовых эмбеддингов."""
    def __init__(self, latent_dir: str, text_dir: str):
        self.latent_dir = latent_dir
        self.text_dir = text_dir
        
        # Снайперская сборка и сортировка списков для синхронизации индексов
        self.files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])
        
        # Проверка герметичности трюмов данных
        text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.pt')])
        if len(self.files) != len(text_files):
            raise RuntimeError(f"[АВАРИЯ] Рассинхронизация кэша! Латентных файлов: {len(self.files)}, Текстовых: {len(text_files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # 1. Извлечение латентного тензора VAE (Строго 1, 16, 128, 128)
        latent_path = os.path.join(self.latent_dir, file_name)
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        if latent.shape != torch.Size([1, 16, 128, 128]):
            raise ValueError(f"[FAIL] Битый размер латента {file_name}: {latent.shape}")
            
        # Убираем фантомную батч-ось [1, 16, 128, 128] -> [16, 128, 128]
        latent = latent.squeeze(0)

        # 2. Извлечение скрытых состояний текста по истинным ключам
        text_path = os.path.join(self.text_dir, file_name)
        text_dict = torch.load(text_path, map_location="cpu", weights_only=True)
        
        if not isinstance(text_dict, dict):
            raise TypeError(f"[FAIL] Кэш текста {file_name} не является словарем!")
            
        clip_hidden = text_dict.get("clip_hidden")
        t5_hidden = text_dict.get("t5_hidden")
        
        if clip_hidden is None or t5_hidden is None:
            raise KeyError(f"[FAIL] В контейнере {file_name} отсутствуют маршевые ключи!")

        # Возврат в чистом bfloat16, готовом для инжекции в MMDiT
        return {
            "latent": latent.to(torch.bfloat16),
            "clip_hidden": clip_hidden.to(torch.bfloat16),
            "t5_hidden": t5_hidden.to(torch.bfloat16),
            "file_name": file_name
        }
