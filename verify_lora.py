import os
from safetensors.torch import safe_open

lora_path = r"Z:\flowch\chroma1_mangala_lora.safetensors"

if not os.path.exists(lora_path):
    print(f"❌ тоговый файл LoRA не найден: {lora_path}")
else:
    print(f"📡 нализ испеченной LoRA: {os.path.basename(lora_path)}")
    print(f"⚖️ азмер файла на SSD: {os.path.getsize(lora_path) / (1024*1024):.2f} ")
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"✅ спешно! сего извлечено LoRA-тензоров: {len(keys)}")
            print("\n📋 Структура ключевых матриц весов (ример):")
            for key in keys[:6]:
                tensor_slice = f.get_slice(key)
                print(f"  - {key} | Shape: {tensor_slice.get_shape()} | Dtype: {tensor_slice.get_dtype()}")
    except Exception as e:
        print(f"❌ шибка проверки внутренней структуры: {e}")