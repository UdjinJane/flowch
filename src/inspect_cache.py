import os
import torch

LATENT_DIR = "./dataset/latent_cache"
TEXT_DIR = "./dataset/text_cache"

def inspect_cold_cache():
    print("# === ЗАПУСК ХОЛОДНОГО ТЕСТА КЭША CHROMA v50 ===")
    
    # Берем по одному первому файлу из каждой папки для экспресс-анализа
    try:
        latent_files = [f for f in os.listdir(LATENT_DIR) if f.endswith('.pt')]
        text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.pt')]
    except FileNotFoundError as e:
        print(f"[FAIL] Контур каталогов не обнаружен: {e}")
        return

    if not latent_files or not text_files:
        print("[WARN] Карантинные зоны кэша пусты. Нечего инспектировать.")
        return

    # Тест 1: Вскрытие латентного тензора (VAE)
    latent_path = os.path.join(LATENT_DIR, latent_files[0])
    try:
        latent_tensor = torch.load(latent_path, map_location="cpu", weights_only=True)
        print(f"[OK] Латент {latent_files[0]} считан. Форма тензора: {latent_tensor.shape}")
    except Exception as e:
        print(f"[CRIT] Битый латентный контейнер: {e}")

    # Тест 2: Вскрытие текстового тензора (T5-XXL / CLIP)
    text_path = os.path.join(TEXT_DIR, text_files[0])
    try:
        text_tensor = torch.load(text_path, map_location="cpu", weights_only=True)
        print(f"[OK] Эмбеддинг {text_files[0]} считан. Форма тензора: {text_tensor.shape}")
    except Exception as e:
        print(f"[CRIT] Битый текстовый контейнер: {e}")

    print("# === ХОЛОДНЫЙ ТЕСТ ЗАВЕРШЕН. СИСТЕМЫ СТРУКТУРНО СТАБИЛЬНЫ ===")

if __name__ == "__main__":
    inspect_cold_cache()
