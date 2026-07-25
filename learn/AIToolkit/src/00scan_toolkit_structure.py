import os

def generate_toolkit_map():
    source_dir = r"Z:\AI-Toolkit-Easy-Install\AI-Toolkit"
    # Кладовочка, шаг назад от папки скрипта
    output_dir = r"Z:\flowch\learn\AIToolkit"
    output_file = os.path.join(output_dir, "toolkit_structure_map.txt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Исключаем системный и виртуальный мусор, чтобы не замыливать глаза
    ignored_dirs = {'.git', '.venv', 'venv', '__pycache__', 'output', 'images', 'data'}
    
    print(f"[РАДАР] Начинаю картирование структуры: {source_dir}")
    
    file_list = []
    
    for root, dirs, files in os.walk(source_dir):
        # Фильтруем папки на лету
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            # Нас интересуют только скрипты и конфигурации
            if file.endswith(('.py', '.json', '.yaml', '.yml')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                file_size_kb = os.path.getsize(full_path) / 1024
                file_list.append((rel_path, file_size_kb))
                
    # Сортируем по названию для идеальной читаемости дерева
    file_list.sort(key=lambda x: x[0])
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=====================================================================\n")
        f.write(f"🛰️ КАРТА СТРУКТУРЫ AI-TOOLKIT — КЛАДОВОЧКА V02\n")
        f.write(f"Исходный путь: {source_dir}\n")
        f.write(f"Всего верифицировано целевых файлов: {len(file_list)}\n")
        f.write("=====================================================================\n\n")
        
        for rel_path, size in file_list:
            f.write(f"[{size:8.2f} KB] -> {rel_path}\n")
            
    print(f"[УСПЕХ] Структура запечена в кладовочке: {output_file}")

if __name__ == "__main__":
    generate_toolkit_map()
