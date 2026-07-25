# === БЛОК ДАННЫХ V02 СТАРТ ===
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from config import TrainConfig

class CachedFluxDatasetV02(Dataset):
    def __init__(self):
        print("[ОБТ] Инициализация стерильного отсека данных: Dataset_V02")
        self.samples = []



        if not os.path.exists(TrainConfig.METADATA_PATH):
            print(f"[КРИТ] Манифест не найден по пути: {TrainConfig.METADATA_PATH}")
            return

        with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                img_name = data["file_name"]
                base_name = os.path.splitext(img_name)[0]

                # === СТВОР МОНОЛИТНОГО КЭША V08_LOCAL ===
                # Привязываемся строго к единому монолитному файлу .pt из папки text_cache
                # === ФИКС СУФФИКСА МОНОЛИТА V08_LOCAL ===
                mono_text_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}.pt")
                latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}.pt")

                # === ВРЕМЕННЫЙ ДИАГНОСТИЧЕСКИЙ РАДАР V08 ===
                # Выводим в лог первую строку, чтобы глазами увидеть, что и где ищет скрипт
                if len(self.samples) == 0:
                    print(f"[ОТЛАДКА ПУТЕЙ] Ищу текстовый монолит: {mono_text_path} -> Существует: {os.path.exists(mono_text_path)}")
                    print(f"[ОТЛАДКА ПУТЕЙ] Ищу латенты кадра: {latent_path} -> Существует: {os.path.exists(latent_path)}")
                # === КОНЕЦ РАДАРА ===

                
                if os.path.exists(mono_text_path) and os.path.exists(latent_path):
                    self.samples.append({
                        "mono_text_path": mono_text_path,
                        "latent_path": latent_path,
                        "img_name": img_name
                    })


        print(f"[УСПЕХ] Dataset_V02: Успешно состыковано {len(self.samples)} готовых к плавке кадров.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Вскрываем монолитный словарь T5+CLIP кузнецов V09_FINAL
        cached_dict = torch.load(sample["mono_text_path"], map_location="cpu")
        
        # Достаем истинные ключи из рапорта зонда, срезая ось батча [1, 256, 4096] -> [256, 4096]
        prompt_embeds = cached_dict["t5_hidden"].squeeze(0)
        
        # Автоматическая сборка единичной маски внимания под фиксированную длину кузнецов (256 токенов)
        text_ids_mask = torch.ones((prompt_embeds.shape), dtype=torch.bool)
        
        # Загружаем тяжелые латенты Chroma-материи (513 КБ)
        latents = torch.load(sample["latent_path"], map_location="cpu", weights_only=True).squeeze(0)
        
        # Жесткий контроль геометрии латентов перед отправкой в ядро
        assert latents.shape == (16, 64, 64), f"Unexpected latent shape: {latents.shape}"
        
        return {
            "prompt_embeds": prompt_embeds,
            "text_ids_mask": text_ids_mask,
            "latents": latents
        }


def get_dataloader_v02():
    dataset = CachedFluxDatasetV02()
    if len(dataset) == 0:
        raise ValueError("[КРИТ] Нулевой размер датасета V02! Проверьте кэш.")
    return DataLoader(
        dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
# === БЛОК ДАННЫХ V02 ФИНАЛ ===
