import os
import sys
from pathlib import Path

def print_log(message: str):
    """Бортовой логгер (вывод в стандартный поток с явным указанием UTF-8)"""
    sys.stdout.buffer.write(f"[GEMMA LOG] {message}\n".encode('utf-8'))

def safe_read_file(file_path: Path) -> str:
    """Безопасное чтение файлов с обработкой исключений"""
    if not file_path.exists():
        print_log(f"Внимание: Файл не найден -> {file_path.name}")
        return f"--- ОШИБКА: Файл {file_path.name} отсутствует ---\n\n"
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
        print_log(f"Успешно прочитан: {file_path.name} ({len(content)} символов)")
        return content
    except Exception as e:
        print_log(f"Критическая ошибка чтения {file_path.name}: {str(e)}")
        return f"--- ОШИБКА ЧТЕНИЯ {file_path.name}: {str(e)} ---\n\n"

def pack_source_files(source_dir: Path, file_list: list, output_file: Path):
    """Сборка файлов по списку в единый монолит с разделителями"""
    print_log(f"Начало сборки монолита: {output_file.name}")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"==================================================\n")
        out.write(f"МОНОЛИТ ИСХОДНОГО КОДА: {output_file.name}\n")
        out.write(f"БАЗОВАЯ ДИРЕКТОРИЯ: {source_dir}\n")
        out.write(f"==================================================\n\n")
        
        for file_name in file_list:
            target_path = source_dir / file_name
            out.write(f"\n{'#'*80}\n")
            out.write(f"### ФАЙЛ: {file_name}\n")
            out.write(f"### ПУТЬ: {target_path}\n")
            out.write(f"{'#'*80}\n\n")
            
            file_content = safe_read_file(target_path)
            out.write(file_content)
            out.write("\n\n")
            
    print_log(f"Монолит успешно сохранен. Размер: {output_file.stat().st_size} байт")

def main():
    print_log("Запуск протокола экстракции кода Chroma...")
    
    # 1. Определение путей назначения (Куда складывать)
    dst_src_dir = Path(r"Z:\flowch\learn\AIToolkit\src")
    dst_extracts_dir = Path(r"Z:\flowch\learn\AIToolkit\core_extracts")
    
    # Создаем директории, если их нет
    dst_src_dir.mkdir(parents=True, exist_ok=True)
    dst_extracts_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Определение путей источников (Откуда брать)
    # Корневой репозиторий ИИ-Тулкита на диске Z:
    base_toolkit_path = Path(r"Z:\AI-Toolkit-Easy-Install\AI-Toolkit")
    chroma_base = base_toolkit_path / "extensions_built_in" / "diffusion_models" / "chroma"
    chroma_src = chroma_base / "src"
    
    if not chroma_base.exists():
        print_log(f"Критическая ошибка: Базовая директория Chroma не найдена по пути {chroma_base}")
        sys.exit(1)

    # 3. Улов №1: Корневые файлы расширения Chroma
    chroma_root_files = ["__init__.py", "chroma_model.py", "chroma_radiance_model.py", "pipeline.py"]
    output_root_monolith = dst_extracts_dir / "chroma_extensions.txt"
    pack_source_files(chroma_base, chroma_root_files, output_root_monolith)
    
    # 4. Улов №2: Внутренние файлы ядра из папки src/
    chroma_core_files = ["layers.py", "math.py", "model.py", "radiance.py", "spy.py"]
    output_core_monolith = dst_extracts_dir / "chroma_src.txt"
    pack_source_files(chroma_src, chroma_core_files, output_core_monolith)
    
    # 5. Копирование файла суммаризации (метаданных графа) без изменений
    summary_src = chroma_src / "chroma_core_summary.txt"
    summary_dst = dst_extracts_dir / "chroma_core_summary.txt"
    
    if summary_src.exists():
        try:
            content = summary_src.read_text(encoding='utf-8', errors='replace')
            summary_dst.write_text(content, encoding='utf-8')
            print_log(f"Файл метаданных ядра chroma_core_summary.txt успешно скопирован.")
        except Exception as e:
            print_log(f"Не удалось скопировать chroma_core_summary.txt: {str(e)}")
    else:
        print_log("Файл chroma_core_summary.txt в источнике не обнаружен.")

    print_log("Протокол 'Сачок' успешно выполнен. Все данные в сейфе.")

if __name__ == "__main__":
    main()
