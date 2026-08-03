#---------------- Старт Блока 1 (Динамический сборщик и валидатор манифеста реального кэша мангала)---
import os
import json

DATA_DIR = r"Z:\flowch\dataset"
MNG_DIR = os.path.join(DATA_DIR, "mng_oks_bl")
OUTPUT_FILE = os.path.join(DATA_DIR, "metadata.jsonl")

def build_and_validate_dataset():
    """Сборщик датасета: парсит реальный плацдарм мангала и штампует валидный JSONL."""
    if not os.path.exists(MNG_DIR):
        print(f"[CRIT] Каталог мангала не найден по адресу: {MNG_DIR}")
        return

    print(f"[RUN] Сканирую директорию мангала {MNG_DIR}...")
    
    # Снайперски собираем только валидные пары JPG+TXT
    jpg_files = [f for f in os.listdir(MNG_DIR) if f.lower().endswith('.jpg')]
    valid_count = 0

    print(f"[RUN] Формирую маршевый манифест в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for jpg_name in jpg_files:
            base_name = os.path.splitext(jpg_name)[0]
            txt_name = base_name + ".txt"
            txt_path = os.path.join(MNG_DIR, txt_name)
            
            # Читаем промпт из парного файла, если он существует
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as txt_f:
                    prompt_text = txt_f.read().strip()
            else:
                print(f"[WARN] Пропуск: отсутствует парный txt-файл для {jpg_name}")
                continue

            # Страховочный фильтр на пустые токены против NaN
            if not prompt_text:
                print(f"[WARN] Пропуск битого токена (пустой промпт): {txt_name}")
                continue

            # Запись строго по уставу JSONL (одна строка — один объект)
            entry = {
                "image": jpg_name,
                "text": prompt_text
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"[OK] Контур датасета запечатан. Успешно привязано физических пар: {valid_count}")

if __name__ == "__main__":
    build_and_validate_dataset()
#---------------- Конец Блока 1 -----------------
