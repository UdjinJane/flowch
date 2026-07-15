import os
import json
import sys
# обавляем src в пути поиска модулей, чтобы импортировать config.py
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import TrainConfig

print("--- [Т] Ы Т Т Ы ЫХ ---")
if not os.path.exists(TrainConfig.METADATA_PATH):
    print(f"[Т] анифест отсутствует: {TrainConfig.METADATA_PATH}")
    sys.exit(1)

with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
    for line in f:
        if not line.strip(): 
            continue
        data = json.loads(line)
        img_name = data["file_name"]
        
        # Смотрим, как именно splitext нарезает имя в этой системе
        # ы вытаскиваем строго нулевой индекс кортежа
        base_name = os.path.splitext(img_name)[0]
        
        embed_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_embeds.pt")
        mask_path = os.path.join(TrainConfig.CACHE_TEXT_DIR, f"{base_name}_mask.pt")
        latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}_latents.pt")
        
        print(f"1. мя файла в манифесте:   {img_name}")
        print(f"2. ычисленное базовое имя: {base_name}")
        print(f"3. оиск текста (_embeds): {embed_path} -> {'Т' if os.path.exists(embed_path) else 'ТСТСТТ'}")
        print(f"4. оиск маски (_mask):    {mask_path} -> {'Т' if os.path.exists(mask_path) else 'ТСТСТТ'}")
        print(f"5. оиск латента (_latent): {latent_path} -> {'Т' if os.path.exists(latent_path) else 'ТСТСТТ'}")
        break

print("\n--- СТ С SSD  ---")
print("еальные файлы в text_embeds:", os.listdir(TrainConfig.CACHE_TEXT_DIR)[:2] if os.path.exists(TrainConfig.CACHE_TEXT_DIR) else "апка пуста")
print("еальные файлы в latent_embeds:", os.listdir(TrainConfig.CACHE_LATENT_DIR)[:2] if os.path.exists(TrainConfig.CACHE_LATENT_DIR) else "апка пуста")
