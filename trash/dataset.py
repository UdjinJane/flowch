import os
import json
import torch
from torch.utils.data import Dataset

class ChromaDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"атрица метаданных не найдена по пути: {jsonl_path}")
            
        print(f"📦 [Data] агрузка метаданных из {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"✅ [Data] спешно загружено записей: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # олучаем чистое базовое имя кадра (например, DSC_0465)
        raw_path = item.get("image_path") or item.get("file_name") or ""
        normalized_path = raw_path.replace('/', os.sep).replace('\\', os.sep)
        img_base = os.path.splitext(os.path.basename(normalized_path))[0]
        
        # Собираем пути к кэшу текста и кэшу латентов VAE
        text_cache_path = os.path.join(r"Z:\flowch\dataset\text_cache", f"['{img_base}'].pt")
        latent_cache_path = os.path.join(r"Z:\flowch\dataset\latent_cache", f"['{img_base}'].pt")

        try:
            # а случай если кэшер записал файлы без скобок в имени
            if not os.path.exists(text_cache_path):
                text_cache_path = os.path.join(r"Z:\flowch\dataset\text_cache", f"{img_base}.pt")
            if not os.path.exists(latent_cache_path):
                latent_cache_path = os.path.join(r"Z:\flowch\dataset\latent_cache", f"{img_base}.pt")

            # агружаем готовые тензоры
            text_cache = torch.load(text_cache_path, map_location="cpu", weights_only=True)
            latent_tensor = torch.load(latent_cache_path, map_location="cpu", weights_only=True)
            
            return {
                "latent_values": latent_tensor.squeeze(0), # орма: [16, 128, 128]
                "clip_hidden": text_cache["clip_hidden"].squeeze(0),
                "t5_hidden": text_cache["t5_hidden"].squeeze(0),
                "img_name": img_base
            }
        except Exception as e:
            print(f"⚠️ шибка загрузки кэша для кадра {img_base}: {e}")
            return {
                "latent_values": torch.zeros(16, 128, 128),
                "clip_hidden": torch.zeros(77, 768),
                "t5_hidden": torch.zeros(256, 4096),
                "img_name": img_base
            }