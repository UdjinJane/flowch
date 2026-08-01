import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from pygments import highlight
from pygments.lexers import get_lexer_for_filename
from pygments.formatters import HtmlFormatter

# === КОНФИГУРАЦИЯ КОНТУРА СКАНИРОВАНИЯ ===
TARGET_DIR = "./src"         # Что сканируем
OUTPUT_DIR = "./out_pdf"     # Куда складываем плавки
IGNORE_DIRS = {".venv", "__pycache__", ".git", "out_pdf"}
VALID_EXTENSIONS = {".py", ".txt"}

def scan_project_directory(root_dir):
    """
    Снайперский обход директории. 
    Изолирует целевые файлы от внешнего шума и служебных каталогов.
    """
    target_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Жесткая фильтрация карантинных зон на месте (in-place)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in VALID_EXTENSIONS:
                full_path = os.path.join(root, file)
                target_files.append(full_path)
                
    return target_files

def convert_file_to_pdf(source_path, output_dir):
    """
    Трансформирует сырой исходный код в изолированный векторный PDF-контейнер.
    Сохраняет структуру отступов и кодировку.
    """
    # Вычисляем целевой путь для сохранения плавки
    rel_path = os.path.relpath(source_path)
    safe_name = rel_path.replace(os.sep, "_") + ".pdf"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, safe_name)
    
    # Инициализируем геометрию листа Letter
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36, rightMargin=36,
        topMargin=36, bottomMargin=36
    )
    
    # Настройка стилей холста
    styles = getSampleStyleSheet()
    
    # Моноширинный защищенный стиль для сохранения indents
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',  # Встроенный базовый моноширинный шрифт
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1A1A1A")
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceAfter=12,
        textColor=colors.HexColor("#0066CC")
    )

    story = []
    
    # Заголовок внутри документа для верификации пути
    story.append(Paragraph(f"SOURCE LOG: {rel_path}", header_style))
    story.append(Spacer(1, 10))
    
    try:
        with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Блокировка деформации: Preformatted держит структуру "как есть"
        clean_content = content.replace('\r\n', '\n').replace('\t', '    ')
        story.append(Preformatted(clean_content, code_style))
        
        # Сборка финального слика
        doc.build(story)
        print(f"[OK] Выплавлен PDF контур для: {rel_path} -> {safe_name}")
    except Exception as e:
        print(f"[FAIL] Сбой деквантования файла {rel_path}: {str(e)}")

if __name__ == "__main__":

    print("# === ЗАПУСК АВТОНОМНОГО КОНВЕРТЕРА КОНТЕКСТА v1.0 ===")
    
    # Проверка плацдарма
    if not os.path.exists(TARGET_DIR):
        print(f"[INFO] Создаю пулевой каталог {TARGET_DIR} для исходников...")
        os.makedirs(TARGET_DIR, exist_ok=True)
        
    # Шаг 1: Сканирование
    print(f"[RUN] Сканирую сектор '{TARGET_DIR}' на наличие .py и .txt файлов...")
    files_to_convert = scan_project_directory(TARGET_DIR)
    
    if not files_to_convert:
        print("[WARN] Целевые файлы не обнаружены. Положите исходники в папку /src")
        sys.exit(0)
        
    print(f"[OK] Обнаружено целей для упаковки: {len(files_to_convert)}")
    
    # Шаг 2: Конвертация
    print("[RUN] Запуск маршевого цикла плавки в PDF...")
    for file_path in files_to_convert:
        convert_file_to_pdf(file_path, OUTPUT_DIR)
        
    print("# === КОНТУР УСПЕШНО ЗАВЕРШИЛ РАБОТУ. ВСЕ ПДФ В out_pdf/ ===")
