#---------------- Старт Блока 6 (Кристальный Конвертер Кода v5.1-FINAL с авто-калибровкой длинных строк) ------------
import os
import sys
import html
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# === КОНФИГУРАЦИЯ КОНТУРА СКАНИРОВАНИЯ ===
TARGET_DIR = "./src"
OUTPUT_DIR = "./out_pdf"
IGNORE_DIRS = {".venv", "__pycache__", ".git", "out_pdf"}
VALID_EXTENSIONS = {".py", ".txt"}

try:
    win_font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'cour.ttf')
    if os.path.exists(win_font_path):
        pdfmetrics.registerFont(TTFont('CourierCyr', win_font_path))
        FONT_NAME = 'CourierCyr'
    else:
        FONT_NAME = 'Courier'
        print("[WARN] Системный шрифт Courier New не найден.")
except Exception as e:
    FONT_NAME = 'Courier'
    print(f"[WARN] Ошибка инициализации шрифта: {e}")

def scan_project_directory(root_dir):
    target_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in VALID_EXTENSIONS:
                target_files.append(os.path.join(root, file))
    return sorted(target_files)

def highlight_code_to_clean_xml(source_text):
    """
    Конвертирует код в XML-формат с жесткой нумерацией строк.
    Заменяет пробелы на неразрывные блоки для защиты структуры от Лангольеров.
    """
    xml_output = []
    lines = source_text.split('\n')
    max_num_len = len(str(len(lines)))
    
    for idx, line in enumerate(lines, 1):
        escaped_line = html.escape(line)
        # ЖЕСТКИЙ СИ-ФИКС: Заменяем пробелы на &nbsp;, блокируя автоперенос слов внутри строки кода
        escaped_line = escaped_line.replace(' ', '&nbsp;')
        
        line_num_str = f"{idx:<{max_num_len + 1}}"
        line_num_escaped = line_num_str.replace(' ', '&nbsp;')
        
        formatted_line = (
            f'<font color="#777777">{line_num_escaped}</font>'
            f'<font color="#000000"><b>{escaped_line}</b></font><br/>'
        )
        xml_output.append(formatted_line)
        
    return "".join(xml_output)

def convert_file_to_pdf(source_path, output_dir):
    """
    Трансформирует исходный код в PDF с автоматической калибровкой размера шрифта
    под максимальную длину строк во избежание Си-разрывов слов в рантайме OCR.
    """
    rel_path = os.path.relpath(source_path)
    safe_name = rel_path.replace(os.sep, "_") + ".pdf"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, safe_name)
    
    # Предельно сужаем поля (всего 14 пунктов), расширяя танковую колею под длинный код
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=14, rightMargin=14,
        topMargin=18, bottomMargin=18
    )
    
    try:
        with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        content = content.replace('\r\n', '\n').replace('\t', '    ')
        
        # === АВТО-КАЛИБРОВКА КАЛИБРА ШРИФТА ===
        lines = content.split('\n')
        max_line_len = max(len(l) for l in lines) if lines else 0
        
        # Динамически поджимаем кегль, если кузнецы Метрополии накатали сверхдлинные строки
        if max_line_len > 95:
            target_font_size = 8.0
            target_leading = 11.0
        elif max_line_len > 75:
            target_font_size = 9.5
            target_leading = 13.0
        else:
            target_font_size = 11.5
            target_leading = 15.0
            
        styles = getSampleStyleSheet()
        code_style = ParagraphStyle(
            'DynamicMonochromeStyle',
            parent=styles['Normal'],
            fontName=FONT_NAME,
            fontSize=target_font_size,
            leading=target_leading,
            textColor=colors.black
        )
        
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor("#0066CC")
        )
        
        story = []
        story.append(Paragraph(f"SOURCE LOG: {rel_path} | Max Line: {max_line_len} char (Font: {target_font_size})", header_style))
        story.append(Spacer(1, 4))
        
        highlighted_xml = highlight_code_to_clean_xml(content)
        story.append(Paragraph(highlighted_xml, code_style))
        doc.build(story)
        print(f"[OK] Выплавлен СТЕРИЛЬНЫЙ PDF контур для: {rel_path} (Размер шрифта: {target_font_size})")
    except Exception as e:
        print(f"[FAIL] Сбой конвейера для файла {rel_path}: {str(e)}")

if __name__ == "__main__":
    print("# === ЗАПУСК КРИСТАЛЬНОГО КОНВЕРТЕРА v5.1-FINAL ===")
    files_to_convert = scan_project_directory(TARGET_DIR)
    if not files_to_convert:
        print("[WARN] Целевые файлы в директории ./src не обнаружены.")
        sys.exit(0)
        
    for file_path in files_to_convert:
        convert_file_to_pdf(file_path, OUTPUT_DIR)
    print("# === МОНОХРОМНЫЙ КОНТУР УСПЕШНО ЗАВЕРШИЛ РАБОТУ ===")
#---------------- Конец Блока 6 -----------------
