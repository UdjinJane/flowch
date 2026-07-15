import os
import json
import sys
# одтягиваем конфигурацию путей
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import TrainConfig

print("--- [Т] ТЯ СХЯ СТ  ТСТ ---")
if not os.path.exists(TrainConfig.METADATA_PATH):
    print(f"[Т] анифест отсутствует: {TrainConfig.METADATA_PATH}")
    sys.exit(1)

# 1. итаем все строки текущего манифеста
with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

valid_lines = []
print(f"[Т] Сканируем физическое наличие файлов в папке: {TrainConfig.DATASET_DIR}")

for line in lines:
    if not line.strip(): 
        continue
    data = json.loads(line)
    img_name = data["file_name"]
    img_path = os.path.join(TrainConfig.DATASET_DIR, img_name)
    
    # Сохраняем строку только если картинка физически существует на SSD!
    if os.path.exists(img_path):
        valid_lines.append(line)

print(f"[Т] з {len(lines)} строк манифеста на диске физически найдено: {len(valid_lines)} кадров.")

# 2. ерезаписываем манифест, если обнаружен перекос
if len(valid_lines) < len(lines):
    print("[] бнаружены фантомные строки! ерезаписываем манифест под живой датасет...")
    with open(TrainConfig.METADATA_PATH, "w", encoding="utf-8-sig") as f:
        f.writelines(valid_lines)
    print("[СХ] анифест metadata.jsonl загерметизирован!")
else:
    print("[] анифест полностью соответствует реальным файлам на диске.")
