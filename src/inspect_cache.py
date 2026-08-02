import os
import torch

LATENT_DIR = "./dataset/latent_cache"
TEXT_DIR = "./dataset/text_cache"

def inspect_cold_cache():
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

    # Вскрытие латентного тензора VAE
    latent_path = os.path.join(LATENT_DIR, latent_files[0])
    try:
        latent_tensor = torch.load(latent_path, map_location="cpu", weights_only=True)
        print(f"[OK] Латент {latent_files[0]} считан. Форма: {latent_tensor.shape} | Тип: {latent_tensor.dtype}")
    except Exception as e:
        print(f"[CRIT] Битый латентный контейнер: {e}")

    # Вскрытие текстового контейнера (Снайперский разбор словаря)
    text_path = os.path.join(TEXT_DIR, text_files[0])
    try:
        text_data = torch.load(text_path, map_location="cpu", weights_only=True)
        if isinstance(text_data, dict):
            print(f"[OK] Текстовый кэш {text_files[0]} — это валидный словарь [dict].")
            for key in ["clip", "t5_xxl"]:
                if key in text_data:
                    tensor = text_data[key]
                    if hasattr(tensor, "shape"):
                        print(f"  -> Ключ [{key}]: Считан. Форма: {tensor.shape} | Тип: {tensor.dtype}")
                    else:
                        print(f"  -> [WARN] Ключ [{key}]: Элемент не является тензором!")
                else:
                    print(f"  -> [CRIT] Ключ [{key}] отсутствует в контейнере!")
        else:
            print(f"[WARN] Аномалия: Ожидался словарь, пришел {type(text_data)}")
    except Exception as e:
        print(f"[CRIT] Ошибка десериализации текста: {e}")

    print("# === ИНСПЕКЦИЯ ЗАВЕРШЕНА. ДАННЫЕ ОПОЗНАНЫ ===")

if __name__ == "__main__":
    inspect_cold_cache()
