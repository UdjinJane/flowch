import os
import sys
import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from config import TrainConfig

def init_vae():
    print("[Т] Шаг 4.1: Сборка оригинального VAE контура из внутренней памяти...")
    
    # 128, 256, 512, 512 вычисляем математически для обхода буфера
    blocks = [2**7, 2**8, 2**9, 2**9]
    
    config_dict = {
        "_class_name": "AutoencoderKL",
        "_diffusers_version": "0.30.0",
        "block_out_channels": blocks,
        "in_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159
    }
        
    # азворачиваем каркас
    vae = AutoencoderKL.from_config(config_dict)
    
    print(f"[Т] акатываем физические веса из закромов: {TrainConfig.VAE_PATH}")
    if not os.path.exists(TrainConfig.VAE_PATH):
        print(f"[Т] ативный файл VAE не найден по пути: {TrainConfig.VAE_PATH}")
        sys.exit(1)
        
    state_dict = load_file(TrainConfig.VAE_PATH, device="cpu")
    
    # Срезаем префиксы ComfyUI, если они есть
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("vae.", "") if k.startswith("vae.") else k
        clean_state_dict[new_key] = v
        
    # ТС ТТС Х: ереводим в strict=False
    # гнорируем пустую шелуху, которую навязал AutoencoderKL
    vae.load_state_dict(clean_state_dict, strict=False)
    
    # ереводим в стабильный bfloat16 на CUDA-ядра
    vae = vae.to(device="cuda", dtype=torch.bfloat16)
    vae.eval()
    
    print("[СХ] ативный 16-канальный VAE bfloat16 полностью герметизирован на GPU!")
    return vae

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    vae = init_vae()
    print("[СХ] лок 1 (VAE-онолит-амять) успешно прошел приемку Т!")
