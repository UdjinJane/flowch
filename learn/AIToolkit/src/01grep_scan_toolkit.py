import os

def run_grep_scan():
    source_dir = r"Z:\AI-Toolkit-Easy-Install\AI-Toolkit"
    output_file = r"Z:\flowch\learn\AIToolkit\vae_grep_raw_leak.txt"
    map_file = r"Z:\flowch\learn\AIToolkit\toolkit_structure_map.txt"
    
    # Ключевые маркеры Лодстоновских диверсий
    #keywords = ["vae", "decode", "channels", "post_quant_conv", "fakevae", "block_out_channels"]
    keywords =['optimizer', 'zero_grad', 'backward', 'step']
    
    # Жесткий контур главных подозреваемых
    core_targets = [
        r"extensions_built_in\diffusion_models\chroma\chroma_model.py",
        r"extensions_built_in\diffusion_models\chroma\chroma_radiance_model.py",
        r"toolkit\models\FakeVAE.py",
        r"toolkit\models\flux.py",
        r"toolkit\config_modules.py",
        r"flux_train_ui.py"
    ]
    
    # Динамический добор всех .py скриптов из карты структуры (автомат без заглушек)
    extended_targets = set()
    if os.path.exists(map_file):
        with open(map_file, "r", encoding="utf-8") as f:
            for line in f:
                if "-> " in line and line.strip().endswith(".py"):
                    rel_path = line.split("-> ")[1].strip()
                    if rel_path.startswith("toolkit") or rel_path.startswith("extensions"):
                        extended_targets.add(rel_path)
                        
    # Сводим все цели в единый верифицированный список
    all_targets = sorted(list(set(core_targets).union(extended_targets)))
    
    print(f"[РАДАР] Начинаю сплошной grep-анализ по {len(all_targets)} скриптам тулкита...")
    
    matches_found = 0
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("=====================================================================\n")
        out.write("🛰️ ЛОГ ОБЪЕКТИВНОГО КОНТРОЛЯ VAE — КЛАДОВОЧКА V02\n")
        out.write(f"Целевых ключевых слов для поиска: {keywords}\n")
        out.write("=====================================================================\n\n")
        
        for rel_path in all_targets:
            full_path = os.path.join(source_dir, rel_path)
            if not os.path.exists(full_path):
                continue
                
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    
                file_has_matches = False
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in keywords):
                        if not file_has_matches:
                            out.write(f"\n📁 ФАЙЛ: {rel_path}\n" + "-"*60 + "\n")
                            file_has_matches = True
                        out.write(f"Строка {line_num:4d}: {line.strip()}\n")
                        matches_found += 1
            except Exception as e:
                out.write(f"⚠️ АВАРИЯ ЧТЕНИЯ {rel_path}: {e}\n")
                
    print(f"[УСПЕХ] Поиск завершен. Зафиксировано совпадений: {matches_found}")
    print(f"[УСПЕХ] Лог объективного контроля запечен: {output_file}")

if __name__ == "__main__":
    run_grep_scan()
