import os
import torch

def audit_full_environment():
    base_dir = "Z:\\flowch"
    print("=" * 70)
    print(f"[ОБЪЕКТИВНЫЙ КОНТРОЛЬ] СКАНИРОВАНИЕ КОНТУРА ДАННЫХ: {base_dir}")
    print("=" * 70)
    
    # 1. Сканирование дерева файлов (без тяжелого мусора .venv и .git)
    print("\n[КАРТА ДИРЕКТОРИЙ И ВЕСА СНАРЯДОВ]:")
    for root, dirs, files in os.walk(base_dir):
        if any(x in root for x in [".venv", ".git", "__pycache__", "models"]):
            continue
        rel_path = os.path.relpath(root, base_dir)
        indent = "  " * (0 if rel_path == "." else rel_path.count(os.sep) + 1)
        folder = os.path.basename(root) if rel_path != "." else "flowch (корень)"
        print(f"{indent}[Папка] {folder}")
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp) / 1024
            print(f"{indent}  └─ {f} ({sz:.1f} KB)")

    # 2. Послойный вскрытие структуры PT-кэша текста
    text_cache_dir = os.path.join(base_dir, "cache", "text_embeds")
    print(f"\n[ПРОВЕРКА СТРУКТУРЫ КЭША ТЕКСТА]: {text_cache_dir}")
    if os.path.exists(text_cache_dir):
        pt_files = [f for f in os.listdir(text_cache_dir) if f.endswith(".pt")]
        if pt_files:
            sample_file = os.path.join(text_cache_dir, pt_files[0])
            print(f"  -> Обнаружено .pt файлов: {len(pt_files)}")
            print(f"  -> Зондируем первый снаряд: {pt_files[0]}")
            try:
                data = torch.load(sample_file, map_location="cpu")
                if isinstance(data, dict):
                    print("  -> Тип данных: СЛОВАРЬ (dict)")
                    for k, v in data.items():
                        if hasattr(v, "shape"):
                            print(f"     └─ Ключ: '{k}' | Shape: {list(v.shape)} | dtype: {v.dtype}")
                        else:
                            print(f"     └─ Ключ: '{k}' | Тип: {type(v)}")
                elif hasattr(data, "shape"):
                    print(f"  -> Тип данных: ИЗОЛИРОВАННЫЙ ТЕНЗОР | Shape: {list(data.shape)} | dtype: {data.dtype}")
            except Exception as e:
                print(f"  [КРИТ] Сбой зондирования структуры файла: {e}")
        else:
            print("  [⚠] .pt файлы отсутствуют!")
    else:
        print("  [КРИТ] Каталог кэша текста не обнаружен по данному пути!")
    print("=" * 70)

if __name__ == "__main__":
    audit_full_environment()
