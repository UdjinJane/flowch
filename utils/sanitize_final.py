import json
import os
import re

JSONL_PATH = r"Z:\flowch\dataset\mng_oks_bl\metadata.jsonl"
TEMP_PATH = r"Z:\flowch\dataset\mng_oks_bl\metadata_clean.jsonl"

def deep_clean(text: str) -> str:
    # 1. ыжигаем любые вариации отрицаний (no ..., no visible ...) и запятую перед ними
    text = re.sub(r",\s*no\s+[\w\s]+", "", text, flags=re.IGNORECASE)
    
    # 2. даляем круглые скобки, но оставляем текст внутри них
    text = text.replace("(", " ").replace(")", " ")
    
    # 3. бираем случайные двоеточия, точки, тире
    text = text.replace(":", "").replace("-", "").replace(".", "")
    
    # 4. Схлопываем множественные запятые и пробелы
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+", " ", text)
    
    # 5. инальная полировка краев
    text = text.strip().strip(",")
    
    # арантируем идеальный пробел после каждой запятой
    parts = [p.strip() for p in text.split(",")]
    return ", ".join(parts)

def run_sanitizer():
    print("[Т] апуск супер-валидатора с защитой от UTF-8 BOM...")
    cleaned_lines = []
    
    if not os.path.exists(JSONL_PATH):
        print(f"[Ш] анифест не найден: {JSONL_PATH}")
        return

    # СЬ utf-8-sig ТЫ ТТС СЬ BOM СТ WINDOWS
    with open(JSONL_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # рокачиваем текст через регулярки
            data["text"] = deep_clean(data["text"])
            cleaned_lines.append(json.dumps(data, ensure_ascii=False))
            
    with open(TEMP_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines) + "\n")
        
    # аменяем старый файл чистым монолитом
    if os.path.exists(JSONL_PATH):
        os.remove(JSONL_PATH)
    os.rename(TEMP_PATH, JSONL_PATH)
    print("[СХ] лубокая фильтрация завершена. BOM удален, отрицания выжжены!")

if __name__ == "__main__":
    run_sanitizer()
