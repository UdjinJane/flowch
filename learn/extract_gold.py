import os
import sys
import shutil

def wash_gold():
    print("[Т] Магистральный запуск промывочного шлюза: extract_gold")
    
    # Absolute coordinates
    ao_root = "Z:\\flowch\\learn\\ao"
    gold_mine = "Z:\\flowch\\learn"
    
    # 1. Проверяем физическое наличие контрабандного отсека
    if not os.path.exists(ao_root):
        print(f"[КРИТ] Отсек {ao_root} не найден на диске! Золотишко украли до нас.")
        sys.exit(1)
        
    # Целевые маршруты к золотым ядрам внутри TorchAO
    optim_src = os.path.join(ao_root, "torchao", "optim")
    sparsity_src = os.path.join(ao_root, "torchao", "sparsity")
    
    # Стерильные шлюзы назначения в трюме learn
    optim_dest = os.path.join(gold_mine, "extracted_optim")
    sparsity_dest = os.path.join(gold_mine, "extracted_sparsity")
    
    # 2. Выковыриваем квантованные оптимизаторы
    if os.path.exists(optim_src):
        print("[ОТК] Извлекаю нативные ядра квантования оптимизаторов...")
        shutil.rmtree(optim_dest, ignore_errors=True)
        # Фикс: убран некорректный аргумент ignore_errors
        shutil.copytree(optim_src, optim_dest)
        print(f"[УСПЕХ] Модули оптимизации намыты в: {optim_dest}")
    else:
        print("[ВНИМАНИЕ] Папка оригинальных оптимизаторов optim не найдена!")

    # 3. Выковыриваем алгоритмы разреженности
    if os.path.exists(sparsity_src):
        print("[ОТК] Извлекаю чертежи разреженности тензоров...")
        shutil.rmtree(sparsity_dest, ignore_errors=True)
        # Фикс: убран некорректный аргумент ignore_errors
        shutil.copytree(sparsity_src, sparsity_dest)
        print(f"[УСПЕХ] Модули разреженности весов намыты в: {sparsity_dest}")
    else:
        print("[ВНИМАНИЕ] Папка оригинальной разреженности sparsity не найдена!")

    # 4. Запекаем краткую инженерную инструкцию
    cheatsheet_path = os.path.join(gold_mine, "gold_usage_cheatsheet.md")
    cheatsheet_content = """# ИНСТРУКЦИЯ ПО ИНЖЕКЦИИ НАМЫТОГО ЗОЛОТА TORCHAO

## 1. Квантование состояний AdamW (Снижение полки VRAM на 2-3 ГБ)
Вместо стандартного прожорливого AdamW в `train_engine_v02.py` теперь можно подключить 8-битное ядро:
```python
import sys
sys.path.append("Z:\\\\flowch\\\\learn")
from extracted_optim import AdamW8bit

# Снайперская замена маршевого узла оптимизатора:
optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE)
```

## 2. Агрессивный CPU-оффлоад градиентов (Полная защита от OOM при батче > 1)
```python
import torch
sys.path.append("Z:\\\\flowch\\\\learn")
from extracted_optim import CPUOffloadOptimizer

optimizer = CPUOffloadOptimizer(
    trainable_params, 
    torch.optim.AdamW, 
    lr=TrainConfig.LEARNING_RATE, 
    fused=True
)
```
"""
    with open(cheatsheet_path, "w", encoding="utf-8") as f:
        f.write(cheatsheet_content.strip())
        
    print(f"[УСПЕХ] Инженерная шпаргалка запечена по адресу: {cheatsheet_path}")
    print("[ОТК] Сепарирование золота завершено. Контур герметичен!")

if __name__ == "__main__":
    wash_gold()
