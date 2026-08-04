import os
import sys
import html
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Подсистемы шрифтов
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Подсистемы подсветки синтаксиса
from pygments import lex
from pygments.lexers import get_lexer_for_filename
from pygments.token import Token

# === КОНФИГУРАЦИЯ КОНТУРА СКАНИРОВАНИЯ ===
TARGET_DIR = "./src" 
OUTPUT_DIR = "./out_pdf" 
IGNORE_DIRS = {".venv", "__pycache__", ".git", "out_pdf"}
VALID_EXTENSIONS = {".py", ".txt"}

# === СВЕРХКОНТРАСТНАЯ НЕОНОВАЯ ПАЛИТРА ДЛЯ ЛАЗЕРНОЙ ОПТИКИ КЭПА ===
COLOR_MAP = {
    Token.Keyword: "#0000FF",        # 100% Синий (def, import, return)
    Token.Name.Function: "#FF00FF", # Насыщенная Маджента (имена функций)
    Token.String: "#8B0000",        # Темно-красный Кровяной (строки и пути)
    Token.Comment: "#006400",       # Глубокий Хвойно-зеленый (комментарии)
    Token.Number: "#FF4500",        # Огненно-рыжий (числа, мантиссы, ранги)
    Token.Operator: "#000000",      # Абсолютно Черный (знаки =, +, -, лоджики)
    Token.Name.Builtin: "#4B0082"   # Насыщенный Индиго (print, len, isinstance)
}



# Инициализация кириллического моноширинного шрифта
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
    return target_files


def highlight_code_to_clean_xml(source_text, filename):
    """
    Конвертирует код в XML-формат ReportLab.
    Заменяет переводы строк на <br/>, а пробелы на &nbsp; 
    для сохранения идеальной структуры кода.
    """
    try:
        lexer = get_lexer_for_filename(filename)
    except Exception:
        # Для .txt файлов просто экранируем и размечаем структуру
        escaped = html.escape(source_text)
        return escaped.replace('\n', '<br/>').replace(' ', '&nbsp;')

    xml_output = []
    
    # Гоним код через лексер pygments
    for token_type, value in lex(source_text, lexer):
        # 1. Экранируем спецсимволы (<, >, &), чтобы XML не падал
        escaped_value = html.escape(value)
        
        # 2. Форматируем переносы и отступы под стандарты Paragraph
        escaped_value = escaped_value.replace('\n', '<br/>').replace(' ', '&nbsp;')
        
        # 3. Ищем цвет для токена
        color = None
        for t_type, hex_color in COLOR_MAP.items():
            if token_type in t_type:
                color = hex_color
                break
        
        if color:
            xml_output.append(f'<font color="{color}">{escaped_value}</font>')
        else:
            xml_output.append(escaped_value)
            
    return "".join(xml_output)


def convert_file_to_pdf(source_path, output_dir):
    """
    Трансформирует код в PDF с идеальным сохранением структуры переносов,
    отступов, цвета и полной поддержкой русского языка.
    """
    rel_path = os.path.relpath(source_path)
    safe_name = rel_path.replace(os.sep, "_") + ".pdf"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, safe_name)
    
    # Сужаем поля, чтобы длинные строки реже переносились
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=24, rightMargin=24,
        topMargin=24, bottomMargin=24
    )
    
    styles = getSampleStyleSheet()
    
    # МАКСИМАЛЬНОЕ ПРОСВЕТЛЕНИЕ: Сверхконтраст и расширенный интерлиньяж
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=8.5,               # Крупный калибр для легкого чтения
        leading=12.0,               # Мощный вертикальный шаг — строки больше не слипнутся
        textColor=colors.black      # Радикально черный пигмент текста базы
    )


    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        spaceAfter=10,
        textColor=colors.HexColor("#0066CC")
    )

    story = []
    story.append(Paragraph(f"SOURCE LOG: {rel_path}", header_style))
    story.append(Spacer(1, 5))
    
    try:
        with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Унифицируем табы и переносы Windows
        content = content.replace('\r\n', '\n').replace('\t', '    ')
        
        # Генерируем подсвеченный XML-абзац
        highlighted_xml = highlight_code_to_clean_xml(content, source_path)
        
        # Скармливаем чистый XML в Paragraph
        story.append(Paragraph(highlighted_xml, code_style))
        
        doc.build(story)
        print(f"[OK] Выплавлен ИДЕАЛЬНЫЙ PDF контур для: {rel_path}")
    except Exception as e:
        print(f"[FAIL] Сбой деквантования файла {rel_path}: {str(e)}")


if __name__ == "__main__":
    print("# === ЗАПУСК КРИСТАЛЬНОГО КОНВЕРТЕРА v4.0 ===")
    files_to_convert = scan_project_directory(TARGET_DIR)
    
    if not files_to_convert:
        print("[WARN] Целевые файлы не обнаружены.")
        sys.exit(0)
        
    for file_path in files_to_convert:
        convert_file_to_pdf(file_path, OUTPUT_DIR)
    print("# === КОНТУР УСПЕШНО ЗАВЕРШИЛ РАБОТУ ===")
