#-------------------- Полный блок: Обновленный скрипт src\inspect_cache.py --------------------
import os
import torch

LATENT_DIR = "./dataset/latent_cache"
TEXT_DIR = "./dataset/text_cache"

def inspect_cold_cache():
    """
    Авторитарная инспекция кэша Chroma v50 [10.1].
    Синхронизирована с истинными маршевыми ключами пульта управления датасетом.
    """
    print("# === ЗАПУСК АВТОРИТАРНОЙ ИНСПЕКЦИИ КЭША CHROMA v50 ===")
    try:
        latent_files = [f for f in os.listdir(LATENT_DIR) if f.endswith('.pt')]
        text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.pt')]
    except FileNotFoundError as e:
        print(f"[FAIL] Контур каталогов разрушен: {e}")
        return

    if not latent_files or not text_files:
        print("[WARN] Карантинные зоны пусты.")
        return

    # 1. Вскрытие и проверка латентного тензора VAE
    latent_path = os.path.join(LATENT_DIR, latent_files[0])
    try:
        latent_tensor = torch.load(latent_path, map_location="cpu", weights_only=True)
        print(f"[OK] Латент {latent_files[0]} считан. Форма: {latent_tensor.shape} | Тип: {latent_tensor.dtype}")
    except Exception as e:
        print(f"[CRIT] Битый латентный контейнер: {e}")

    # 2. Вскрытие текстового контейнера (Синхронизированные ключи)
    text_path = os.path.join(TEXT_DIR, text_files[0])
    try:
        text_data = torch.load(text_path, map_location="cpu", weights_only=True)
        if isinstance(text_data, dict):
            print(f"[OK] Текстовый кэш {text_files[0]} — это валидный словарь [dict].")
            
            # Истинные ключи согласно спецификации пульта управления ChromaDataset
            for key in ["clip_hidden", "t5_hidden"]:
                if key in text_data:
                    tensor = text_data[key]
                    if hasattr(tensor, "shape"):
                        print(f" -> Ключ [{key}]: Считан. Форма: {tensor.shape} | Тип: {tensor.dtype}")
                    else:
                        print(f" -> [WARN] Ключ [{key}]: Элемент не является тензором!")
                else:
                    print(f" -> [CRIT] Ключ [{key}] отсутствует в контейнере! Требуется перегенерация кэша.")
        else:
            print(f"[WARN] Аномалия: Ожидался словарь, пришел {type(text_data)}")
    except Exception as e:
        print(f"[CRIT] Ошибка десериализации текста: {e}")
        
    print("# === ИНСПЕКЦИЯ ЗАВЕРШЕНА. ДАННЫЕ ОПОЗНАНЫ ===")

if __name__ == "__main__":
    inspect_cold_cache()
#-------------------- Окончание блока инспектора --------------------
