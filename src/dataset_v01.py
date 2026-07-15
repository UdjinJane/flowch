import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from config import TrainConfig

class CachedFluxDatasetV01(Dataset):
    def __init__(self):
        print("[Т] нициализация версионированного отсека данных: Dataset_V01")
        self.samples = []
        
        if not os.path.exists(TrainConfig.METADATA_PATH):
            print(f"[Т] анифест не найден по пути: {TrainConfig.METADATA_PATH}")
            return

        # итаем наш эталонный манифест
        with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                img_name = data["file_name"]
                
                # СТ С: ерем строго [0] элемент кортежа (только имя без .JPG)!
                base_name = os.path.splitext(img_name)[0]
                
                # рописываем строго очищенные пути к тензорам на SSD
                embed_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_embeds.pt")
                mask_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_mask.pt")
                latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}_latents.pt")
                
                # роверяем наличие тройки файлов
                if os.path.exists(embed_path) and os.path.exists(mask_path) and os.path.exists(latent_path):
                    self.samples.append({
                        "embed_path": embed_path,
                        "mask_path": mask_path,
                        "latent_path": latent_path,
                        "img_name": img_name
                    })
                    
        print(f"[СХ] Dataset_V01: спешно состыковано {len(self.samples)} готовых к плавке кадров.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # однимаем тензоры в оперативную память напрямую (в CPU)
        prompt_embeds = torch.load(sample["embed_path"], map_location="cpu").squeeze(0)
        text_ids_mask = torch.load(sample["mask_path"], map_location="cpu").squeeze(0)
        latents = torch.load(sample["latent_path"], map_location="cpu").squeeze(0)
        
        return {
            "prompt_embeds": prompt_embeds,
            "text_ids_mask": text_ids_mask,
            "latents": latents
        }

def get_dataloader_v01():
    dataset = CachedFluxDatasetV01()
    if len(dataset) == 0:
        raise ValueError("[Т] улевой размер датасета! Тензоры на диске не соответствуют манифесту.")
    return DataLoader(
        dataset, 
        batch_size=TrainConfig.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )

if __name__ == "__main__":
    try:
        loader = get_dataloader_v01()
        batch = next(iter(loader))
        print("--- [Т] ТСТ С ЫХ V01 СТЬ  ---")
        print(f"орма промпта: {batch['prompt_embeds'].shape}")
        print(f"орма маски:   {batch['text_ids_mask'].shape}")
        print(f"орма латента: {batch['latents'].shape}")
    except Exception as e:
        print(f"--- [Я СТ СТ] ---")
        print(f"ричина отказа: {e}")
