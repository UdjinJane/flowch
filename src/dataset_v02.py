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
                
                # Извлекаем чистое базовое имя (например, DSC_0465)
                base_name = os.path.splitext(img_name)[0]
                
                embed_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_embeds.pt")
                mask_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_mask.pt")
                latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}_latents.pt")
                
                if os.path.exists(embed_path) and os.path.exists(mask_path) and os.path.exists(latent_path):
                    self.samples.append({
                        "embed_path": embed_path,
                        "mask_path": mask_path,
                        "latent_path": latent_path,
                        "img_name": img_name
                    })
                    
        print(f"[УСПЕХ] Dataset_V02: Успешно состыковано {len(self.samples)} готовых к плавке кадров.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # СНАЙПЕРСКИЙ ФИКС: Включаем weights_only=True для тотальной зачистки предупреждений в консоли
        prompt_embeds = torch.load(sample["embed_path"], map_location="cpu", weights_only=True).squeeze(0)
        text_ids_mask = torch.load(sample["mask_path"], map_location="cpu", weights_only=True).squeeze(0)
        latents = torch.load(sample["latent_path"], map_location="cpu", weights_only=True).squeeze(0)
        
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
