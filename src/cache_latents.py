 -*- coding: utf-8 -*-
import os
import sys
import torch
import json
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.dataset import ChromaDataset

def cache_vae_latents():
    print("📦 Старт 100% Т фазы кэширования латентов VAE...")
    vae_path = r"Z:\AiModels\models\vae\ae.safetensors"
    jsonl_path = r"Z:\flowch\metadata.jsonl"
    latent_dir = r"Z:\flowch\dataset\latent_cache"
    
    os.makedirs(latent_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🧠 Сборка каркаса Flux/Chroma VAE из локальной конфигурации...")
    
    # естко объявляем оригинальную топологию Flux VAE (327 )
    # то избавляет нас от скачивания config.json из закрытых репозиториев HF
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        latent_channels=16, #  Flux/Chroma 16 латентных каналов
        norm_num_groups=32
    )

    print(f"📂 агрузка локальных весов: {os.path.basename(vae_path)}")
    vae_sd = load_file(vae_path)
    
    # апим веса diffusers под стандартные ключи safetensors VAE
    vae.load_state_dict(vae_sd, strict=False)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae.eval()
    print("✅ окальный VAE успешно запечен на GPU без интернета!")

    dataset = ChromaDataset(jsonl_path=jsonl_path, target_res=1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"🔄 одирование {len(dataset)} кадров мангала в компактные латенты...")
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            pixels = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            
            # звлекаем чистое имя файла из батча
            img_name_raw = batch["img_name"][0] if isinstance(batch["img_name"], list) else batch["img_name"]
            
            # ереводим пиксели в латентное пространство
            posterior = vae.encode(pixels).latent_dist
            latents = posterior.sample() * 0.18215  # асштабирование Flux/Chroma
            
            out_path = os.path.join(latent_dir, f"{img_name_raw}.pt")
            torch.save(latents.cpu().to(torch.bfloat16), out_path)
            print(f"  [+] спешно упакован: {img_name_raw}.pt | Shape: {list(latents.shape)}")

    print(f"🎉 эширование латентов полностью завершено! аза на SSD: {latent_dir}")

if __name__ == "__main__":
    cache_vae_latents()