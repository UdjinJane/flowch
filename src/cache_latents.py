import os
import sys
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from config import TrainConfig

def init_vae():
    print(f"[Т] Шаг 4.1: Снайперская сборка FLUX-VAE контура: {TrainConfig.VAE_PATH}")
    if not os.path.exists(TrainConfig.VAE_PATH):
        print("[Т] айл VAE отсутствует в закромах!")
        sys.exit(1)
        
    # ТС С: дален spatial_dims, параметры строго по СТу diffusers
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        block_out_channels=[128, 256, 512],
        layers_per_block=2,
        ch_mult=[1, 2, 4], 
        norm_num_groups=32,
        sample_size=1024,
        scaling_factor=0.3611,
        shift_factor=0.1159
    )
    
    # звлекаем чистые веса и накатываем их на каркас
    state_dict = load_file(TrainConfig.VAE_PATH, device="cpu")
    
    # трезаем префиксы, если ComfyUI перепаковал ключи
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("vae.", "") if k.startswith("vae.") else k
        clean_state_dict[new_key] = v
        
    vae.load_state_dict(clean_state_dict, strict=True)
    
    # ереводим в стабильный bfloat16 на CUDA
    vae = vae.to(device="cuda", dtype=torch.bfloat16)
    vae.eval()
    
    print("[СХ] 16-канальный визуальный отсек VAE bfloat16 полностью герметизирован на GPU!")
    return vae

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    vae = init_vae()
    print("[СХ] лок 1 (VAE-онтур) прошел дефектоскопию.")
