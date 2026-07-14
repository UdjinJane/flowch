import os
from safetensors.torch import safe_open

checkpoint_path = r"Z:\AiModels\models\checkpoints\chroma1\Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors"

with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    
    print("🔎 щем блоки трансформера (double/single blocks)...")
    blocks = [k for k in keys if "block" in k.lower()]
    print(f"сего слоев с упоминанием 'block': {len(blocks)}")
    
    # ыводим примеры из середины и конца списка весов
    print("\n📋 Структура double_blocks (ример):")
    db_samples = [k for k in blocks if "double_block" in k.lower()][:8]
    for k in db_samples:
        t = f.get_slice(k)
        print(f"  - {k} | Shape: {t.get_shape()} | Dtype: {t.get_dtype()}")
        
    print("\n📋 Структура single_blocks (ример):")
    sb_samples = [k for k in blocks if "single_block" in k.lower()][:8]
    for k in sb_samples:
        t = f.get_slice(k)
        print(f"  - {k} | Shape: {t.get_shape()} | Dtype: {t.get_dtype()}")
