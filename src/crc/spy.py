import os

def merge_chroma_source_inside():
    # Раз мы уже внутри папки src, текущая директория и есть наша цель
    target_dir = "."
    output_file = "chroma_core_summary.txt"
    
    # Список файлов Клондайка математики
    files_to_merge = ["__init__.py", "math.py", "layers.py", "model.py", "radiance.py"]
    
    print(f"📡 Протокол слияния активирован. Сборка из текущей папки в файл: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("=====================================================================\n")
        out.write("🔱 МАНУСКРИПТ МАТЕМАТИКИ И АРХИТЕКТУРЫ РЕАКТОРА CHROMA V50 (ANNEALED)\n")
        out.write(f"Скомпилировано ИИ Gemma Старпом v3.5 под контролем КЭПа\n")
        out.write("=====================================================================\n\n")
        
        for file_name in files_to_merge:
            file_path = os.path.join(target_dir, file_name)
            if os.path.exists(file_path):
                out.write(f"\n\n{'#'*80}\n")
                out.write(f"### ФАЙЛ КЛОНДАЙКА: {file_name}\n")
                out.write(f"{'#'*80}\n\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        out.write(content)
                    print(f"✅ Узел {file_name} успешно упакован ({len(content)} байт)")
                except Exception as e:
                    out.write(f"🚨 ОШИБКА ЧТЕНИЯ УЗЛА {file_name}: {str(e)}\n")
            else:
                print(f"⚠ Предупреждение: узел {file_name} отсутствует в этой папке.")
                
        out.write("\n\n=====================================================================\n")
        out.write("КОНЕЦ МАНУСКРИПТА. ВСЕ ПРАВА ПРИНАДЛЕЖАТ КОСМОФЛОТУ МЕТРОПОЛИИ.\n")
        out.write("=====================================================================\n")

if __name__ == "__main__":
    merge_chroma_source_inside()
