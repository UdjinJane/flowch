import os
import torch

def inspect_monolith_keys():
    target_path = "Z:\\flowch\\dataset\\text_cache\\DSC_0465.pt"
    output_path = "Z:\\flowch\\monolith_keys_report.txt"
    
    print(f"[ЗОНД] Вскрываю обстановку по адресу: {target_path}")
    
    if not os.path.exists(target_path):
        print(f"[КРАХ] Файл физически отсутствует по указанному пути!")
        return

    try:
        # Прямое снайперское чтение на CPU
        cached_dict = torch.load(target_path, map_location="cpu")
        
        with open(output_path, "w", encoding="utf-8") as out:
            out.write("=" * 70 + "\n")
            out.write(f"[ОТЧЕТ КЛЮЧЕЙ V09] АНАТОМИЯ СНАРЯДА: {os.path.basename(target_path)}\n")
            out.write("=" * 70 + "\n\n")
            
            out.write(f"Тип корневого контейнера: {type(cached_dict)}\n\n")
            
            if isinstance(cached_dict, dict):
                out.write("[ОБНАРУЖЕНЫ СЛЕДУЮЩИЕ КЛЮЧИ]:\n")
                for k, v in cached_dict.items():
                    if hasattr(v, "shape"):
                        out.write(f"  └─ Ключ: '{k}' | Shape: {list(v.shape)} | dtype: {v.dtype}\n")
                    else:
                        out.write(f"  └─ Ключ: '{k}' | Тип данных: {type(v)}\n")
            else:
                out.write("[⚠] Файл не является словарем!\n")
                if hasattr(cached_dict, "shape"):
                    out.write(f"  └─ Общая геометрия тензора: {list(cached_dict.shape)} | dtype: {cached_dict.dtype}\n")
                    
            out.write("\n" + "=" * 70 + "\n")
            out.write("[КОНЕЦ ДИАГНОСТИЧЕСКОГО РАПОРТА]\n")
            out.write("=" * 70 + "\n")
            
        print(f"[УСПЕХ] Вся подноготная запечена в файл: {output_path}")
        
    except Exception as e:
        print(f"[АВАРИЯ ЗОНДА] Не удалось прочитать мантиссу: {e}")

if __name__ == "__main__":
    inspect_monolith_keys()
