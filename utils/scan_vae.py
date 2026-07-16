import torch
from safetensors.torch import load_file

VAE_PATH = r"Z:\AiModels\models\vae\ae.safetensors"

print("--- [Т] Ы  Т VAE ---")
try:
    sd = load_file(VAE_PATH, device="cpu")
    
    # 1. змеряем геометрию скрытых каналов
    if "encoder.conv_out.weight" in sd:
        print(f"encoder.conv_out.weight shape: {sd['encoder.conv_out.weight'].shape}")
    if "decoder.conv_in.weight" in sd:
        print(f"decoder.conv_in.weight shape:  {sd['decoder.conv_in.weight'].shape}")
        
    # 2. ычисляем количество блоков масштабирования по ключам
    down_blocks = sorted(list(set([int(k.split(".")[2]) for k in sd.keys() if k.startswith("encoder.down.")])))
    up_blocks = sorted(list(set([int(k.split(".")[2]) for k in sd.keys() if k.startswith("decoder.up.")])))
    
    print(f"ндексы down-блоков в энкодере: {down_blocks} (сего: {len(down_blocks)})")
    print(f"ндексы up-блоков в декодере:   {up_blocks} (сего: {len(up_blocks)})")

except Exception as e:
    print(f"[Т] шибка сканирования файла весов: {e}")
