import json
import os
import re

# онстанты путей согласно протоколу Z:\flowch
DATASET_DIR = r"Z:\flowch\dataset\mng_oks_bl"
OUTPUT_JSONL = os.path.join(DATASET_DIR, "metadata.jsonl")

# ерный список фраз-отрицаний, которые калечат эмбеддинги T5
FORBIDDEN_PATTERNS = [
    r",\s*no visible meat or smoke",
    r",\s*no meat present",
    r",\s*no temporary elements",
    r",\s*no rust patinas",
    r",\s*no soot stains"
]

def sanitize_caption(text: str) -> str:
    # 1. ырезаем все фантомные отрицания по черному списку
    for pattern in FORBIDDEN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 2. Санитария синтаксиса: убираем точки в конце, случайные двоеточия и двойные запятые
    text = text.replace(":", "").replace("-", "").replace(".", "")
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip().strip(",")

def build_metadata():
    print("[Т] апуск конвейера очистки и сборки метаданных...")
    manifest_lines = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"[Ш] иректория датасета не найдена: {DATASET_DIR}")
        return

    # Сканируем директорию датасета
    for file_name in os.listdir(DATASET_DIR):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = file_name
            txt_path = os.path.splitext(file_name)[0] + ".txt"
            full_txt_path = os.path.join(DATASET_DIR, txt_path)
            
            if os.path.exists(full_txt_path):
                with open(full_txt_path, "r", encoding="utf-8") as f:
                    raw_caption = f.read()
                
                # рогоняем через Т-очистку
                clean_caption = sanitize_caption(raw_caption)
                
                # ормируем строгую строку jsonl для диффузеров
                entry = {
                    "file_name": img_path,
                    "text": clean_caption
                }
                manifest_lines.append(json.dumps(entry, ensure_ascii=False))
            else:
                print(f"[] ропущен текстовый файл для: {img_path}")

    # апись готового монолита на SSD
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")
        
    print(f"[СХ] айл {OUTPUT_JSONL} успешно запечен. Собрано строк: {len(manifest_lines)}")

if __name__ == "__main__":
    build_metadata()
