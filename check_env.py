import os
import torch

def audit_to_file():
    base_dir = "Z:\\flowch"
    output_file = os.path.join(base_dir, "environment_manifest.txt")
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write(f"[ОБЪЕКТИВНЫЙ КОНТРОЛЬ] ПОЛНЫЙ РЕНТГЕНОВСКИЙ СНИМОК СРЕДЫ: {base_dir}\n")
        out.write("=" * 80 + "\n\n")
        
        # 1. Тотальное сканирование дерева каталогов (исключая тяжелый .venv и .git)
        out.write("[УЗЕЛ 1: КАРТА И ВЕСА СНАРЯДОВ В КАТАЛОГЕ SRC И CACHE]\n")
        for root, dirs, files in os.walk(base_dir):
            if any(x in root for x in [".venv", ".git", "__pycache__", "models"]):
                continue
            rel_path = os.path.relpath(root, base_dir)
            indent = "  " * (0 if rel_path == "." else rel_path.count(os.sep) + 1)
            folder = os.path.basename(root) if rel_path != "." else "flowch (корень)"
            out.write(f"{indent}[Папка] {folder}\n")
            for f in files:
                fp = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fp) / 1024
                    out.write(f"{indent}  └─ {f} ({sz:.2f} KB)\n")
                except Exception as e:
                    out.write(f"{indent}  └─ {f} (Сбой замера веса: {e})\n")
                    
        # 2. Зондирование структуры PyTorch .pt файлов кэша текста
        text_cache_dir = os.path.join(base_dir, "cache", "text_embeds")
        out.write(f"\n\n[УЗЕЛ 2: СПЕКТРАЛЬНОЕ ВСКРЫТИЕ СТРУКТУРЫ PT-КЭША]: {text_cache_dir}\n")
        
        if os.path.exists(text_cache_dir):
            pt_files = sorted([f for f in os.listdir(text_cache_dir) if f.endswith(".pt")])
            if pt_files:
                out.write(f"  -> Всего .pt файлов обнаружено в трюме: {len(pt_files)}\n")
                
                # Берём первые два разнотипных файла для образца (например, embeds и mask)
                samples_to_check = pt_files[:4] if len(pt_files) >= 4 else pt_files
                
                for sample in samples_to_check:
                    sample_file = os.path.join(text_cache_dir, sample)
                    out.write(f"\n  [Зонд снаряда] -> {sample}\n")
                    try:
                        data = torch.load(sample_file, map_location="cpu")
                        if isinstance(data, dict):
                            out.write("     └─ Тип контейнера: СЛОВАРЬ (dict)\n")
                            for k, v in data.items():
                                if hasattr(v, "shape"):
                                    out.write(f"        └─ Ключ: '{k}' | Shape: {list(v.shape)} | dtype: {v.dtype}\n")
                                else:
                                    out.write(f"        └─ Ключ: '{k}' | Тип: {type(v)}\n")
                        elif hasattr(data, "shape"):
                            out.write(f"     └─ Тип контейнера: ИЗОЛИРОВАННЫЙ ТЕНЗОР (torch.Tensor)\n")
                            out.write(f"        └─ Shape: {list(data.shape)} | dtype: {data.dtype}\n")
                        else:
                            out.write(f"     └─ Тип контейнера: НЕСТАНДАРТНЫЙ ОБЪЕКТ | Тип: {type(data)}\n")
                    except Exception as e:
                        out.write(f"     [АВАРИЯ ЗОНДА] Сбой разбора внутренней геометрии: {e}\n")
            else:
                out.write("  [⚠] Внутри каталога .pt файлы отсутствуют!\n")
        else:
            out.write("  [КРИТ] Путь cache\\text_embeds физически не существует на диске!\n")
            
        out.write("\n" + "=" * 80 + "\n")
        out.write("[КОНЕЦ СЛЕДСТВЕННОГО МАНИФЕСТА]\n")
        out.write("=" * 80 + "\n")

    print(f"[УСПЕХ] Манифест среды успешно запечен в файл: {output_file}")

if __name__ == "__main__":
    audit_to_file()
