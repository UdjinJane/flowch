import os
import json

DATA_DIR = "./dataset"
OUTPUT_FILE = os.path.join(DATA_DIR, "metadata.jsonl")

# Наш мини-пакет промптов для калибровки Flux/Chroma
RAW_DATA = [
    {"image": "001.png", "text": "Chroma-fp8 style photorealistic starship bridge, detailed cyber-gothic console"},
    {"image": "002.png", "text": "Chroma-fp8 sci-fi reactor core, glowing plasma, high contrast volumetric light"}
]

def build_and_validate_dataset():
    """Сборщик датасета: проверяет плацдарм и штампует валидный JSONL."""
    os.makedirs(DATA_DIR, exist_ok=True)
    valid_count = 0
    
    print(f"[RUN] Формирую технический датасет в {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in RAW_DATA:
            # Снайперская проверка на пустые значения, чтобы трейнер не выпал в NaN
            if not entry["image"] or not entry["text"].strip():
                print(f"[WARN] Пропуск битого токена: {entry}")
                continue
                
            # Запись строго в формате JSONL (одна строка — один объект)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            valid_count += 1
            
    print(f"[OK] Контур датасета запечатан. Успешно записано строк: {valid_count}")

if __name__ == "__main__":
    # Проверка: если в папке пусто, создаем заглушки для картинок
    build_and_validate_dataset()
    for item in RAW_DATA:
        img_path = os.path.join(DATA_DIR, item["image"])
        if not os.path.exists(img_path):
            with open(img_path, "wb") as empty_img:
                empty_img.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") # Мини-заголовок PNG
            print(f"[INFO] Выплавлен тестовый фантом-файл: {item['image']}")
