# === БЛОК ДАННЫХ V02 СТАРТ: ФРАГМЕНТ 1 ИЗ 3 ===
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from config import TrainConfig

class CachedFluxDatasetV02(Dataset):
    """
    Датасет с RAM-форсажем: загружает кэш в оперативную память при инициализации.
    """
    def __init__(self):
        print("[RAM-ФОРСАЖ] Инициализация стерильного отсека данных: Dataset_V02")
        self.samples = []
        
        if not os.path.exists(TrainConfig.METADATA_PATH):
            print(f"[КРИТ] Манифест не найден: {TrainConfig.METADATA_PATH}")
            return

        # Однократный прожиг диска: затягиваем кэш в ОЗУ шхуны
        with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                base_name = os.path.splitext(data["file_name"])[0]
                
                mono_text_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}.pt")
                latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}.pt")
# === БЛОК ДАННЫХ V02 ФИНАЛ: ФРАГМЕНТ 1 ИЗ 3 ===
# === БЛОК ДАННЫХ V02 СТАРТ: ФРАГМЕНТ 2 ИЗ 3 ===
                if os.path.exists(mono_text_path) and os.path.exists(latent_path):
                    # Потоковое считывание с диска — выполняется строго ОДИН раз на старте
                    cached_dict = torch.load(mono_text_path, map_location="cpu", weights_only=False)
                    
                    # Срезаем ось батча [1, 256, 4096] -> [256, 4096] для T5 эмбеддингов
                    prompt_embeds = cached_dict["t5_hidden"].squeeze(0)
                    
                    # Загрузка тяжелой Chroma-материи (513 КБ)
                    latents = torch.load(latent_path, map_location="cpu", weights_only=True).squeeze(0)
                    
                    # Жесткий контроль геометрии латентов до фиксации в памяти
                    assert latents.shape == (16, 128, 128), f"Unexpected latent shape: {latents.shape}"
                    
                    # Снайперская упаковка чистых тензоров напрямую в оперативную память шхуны
                    self.samples.append({
                        "prompt_embeds": prompt_embeds,
                        "latents": latents,
                        "img_name": data["file_name"]
                    })

        print(f"[УСПЕХ] RAM-Форсаж завершен: {len(self.samples)} образцов успешно запечено в ОЗУ.")
# === БЛОК ДАННЫХ V02 ФИНАЛ: ФРАГМЕНТ 2 ИЗ 3 ===
# === БЛОК ДАННЫХ V02 СТАРТ: ФРАГМЕНТ 3 ИЗ 3 ===
    def __len__(self):
        """Возвращает общий объем запеченных в RAM кадров."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Мгновенная выдача готовых тензоров из ОЗУ шхуны без дискового ввода-вывода (I/O)."""
        return self.samples[idx]

def get_dataloader_v02():
    """Сборка стерильного загрузчика батчей на полной скорости RAM."""
    dataset = CachedFluxDatasetV02()
    if len(dataset) == 0:
        raise ValueError("[КРИТ] Нулевой размер датасета V02! Проверьте манифест и кэш.")
        
    return DataLoader(
        dataset, 
        batch_size=TrainConfig.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
# === БЛОК ДАННЫХ V02 ФИНАЛ: КОНЕЦ МОЗАИКИ ===
