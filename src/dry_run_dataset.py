import sys
import os
import torch
from torch.utils.data import DataLoader

# Проброс путей для работы внутри .venv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.chroma_core import ChromaDataset

LATENT_DIR = "./dataset/latent_cache"
TEXT_DIR = "./dataset/text_cache"

def run_dry_dataset_test():
    print("# === ЗАПУСК СУХОГО ТЕСТА МАРШЕВОГО КОНТУРА ДАННЫХ ===")
    
    try:
        # Инициализация ядра загрузчика
        dataset = ChromaDataset(latent_dir=LATENT_DIR, text_dir=TEXT_DIR)
        print(f"[OK] Dataset собран успешно. Найдено пар данных: {len(dataset)}")
        
        # Сборка DataLoader (Микробатч = 1 для удержания полки VRAM)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True, 
            drop_last=False
        )
        
        # Перехват первого тренировочного пакета (Имитация шага обучения)
        for step, batch in enumerate(dataloader):
            print(f"[СТАДИЯ] Проверка батча №{step + 1}...")
            
            latent = batch["latent"]
            clip_h = batch["clip_hidden"]
            t5_h = batch["t5_hidden"]
            name = batch["file_name"][0]
            
            # Телеметрия геометрии и типов данных
            print(f"  -> Файл: {name}")
            print(f"  -> Латент VAE : Форма {list(latent.shape)} | Тип: {latent.dtype}")
            print(f"  -> CLIP Hidden: Форма {list(clip_h.shape)} | Тип: {clip_h.dtype}")
            print(f"  -> T5 Hidden  : Форма {list(t5_h.shape)} | Тип: {t5_h.dtype}")
            
            # Жесткая верификация на NaN
            if torch.isnan(latent).any() or torch.isnan(clip_h).any() or torch.isnan(t5_h).any():
                print(f"[АВАРИЯ] Обнаружены NaN в пакете {name}!")
                return
                
            print(f"[OK] Батч №{step + 1} прошел сквозной контроль.")
            
            # Ограничиваем сухой прогон двумя батчами для экономии времени
            if step >= 1:
                break
                
        print("# === СУХОЙ ТЕСТ ЗАВЕРШЕН С УСПЕХОМ. СИСТЕМЫ ГОТОВЫ К ПЛАВКЕ ===")
        
    except Exception as e:
        print(f"[КРИТ] Контур данных разрушен на шаге рантайма: {e}")

if __name__ == "__main__":
    run_dry_dataset_test()
