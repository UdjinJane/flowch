import os
import shutil

def extract_core_modules():
    source_base = r"Z:\AI-Toolkit-Easy-Install\AI-Toolkit"
    output_dir = r"Z:\flowch\learn\AIToolkit\core_extracts"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Цели для снайперской вырезки
    targets = {
        "FakeVAE.py": r"toolkit\models\FakeVAE.py",
        "chroma_model.py": r"extensions_built_in\diffusion_models\chroma\chroma_model.py",
        "chroma_pipeline.py": r"extensions_built_in\diffusion_models\chroma\pipeline.py"
    }
    
    print("[РАДАР] Экстракция критических модулей...")
    for name, rel_path in targets.items():
        src = os.path.join(source_base, rel_path)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src):
            shutil.copy_file(src, dst) if hasattr(shutil, 'copy_file') else shutil.copy(src, dst)
            print(f" -> [OK] {name} извлечен в core_extracts")
        else:
            print(f" -> [⚠️] Файл не найден: {rel_path}")
            
    print(f"[УСПЕХ] Выжимка лакун завершена в: {output_dir}")

if __name__ == "__main__":
    extract_core_modules()
