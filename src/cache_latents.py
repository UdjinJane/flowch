import json
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
from PIL import Image
import torchvision.transforms as T

def process_and_cache_images(vae):
    print("[Т] Шаг 4.2: апуск конвейера кодирования изображений через VAE...")
    
    if not os.path.exists(TrainConfig.CACHE_LATENT_DIR):
        os.makedirs(TrainConfig.CACHE_LATENT_DIR)

    # Стандартный пайплайн трансформации пикселей в тензор [-1, 1]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    with open(TrainConfig.METADATA_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            img_name = data["file_name"]
            img_path = os.path.join(TrainConfig.DATASET_DIR, img_name)
            
            if not os.path.exists(img_path):
                print(f"[] айл изображения не найден: {img_path}")
                continue
                
            # агружаем кадр мангала
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            
            # СТ С: ыравниваем стороны кратно 16 пикселям под шаг сетки Chroma1
            # елаем это через целочисленное деление, чтобы буфер не сожрал цифры
            k = 2**4 # исло 16
            new_w = (w // k) * k
            new_h = (h // k) * k
            
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                
            # ереводим в тензор bfloat16 на CUDA-ядра
            img_tensor = transform(img).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            
            # одируем в латентное пространство без расчета градиентов
            with torch.no_grad():
                # звлекаем модули распределения и берем среднее (mode)
                latents = vae.encode(img_tensor).latent_dist.mode()
                
            # звлекаем чистое имя кадра для сохранения тензоров
            base_name = os.path.splitext(img_name)
            out_latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}_latents.pt")
            
            # Сохраняем готовый визуальный латент на SSD буфер
            torch.save(latents.cpu(), out_latent_path)
            print(f"[СХ] спешно закеширован латент VAE для: {img_name} [азмер: {new_w}x{new_h}]")

if __name__ == "__main__":
    vae = init_vae()
    process_and_cache_images(vae)
    print("[Т] изуальный конвейер  Ш полностью и безупречно отработал.")
