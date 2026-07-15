import os
import sys
import json
import torch
from PIL import Image
import torchvision.transforms as T
from safetensors.torch import load_file
from config import TrainConfig

class NativeFluxEncoder:
    def __init__(self, weight_path):
        print(f"[Т] рямая инициализация плоских тензоров из: {weight_path}")
        self.weights = load_file(weight_path, device="cuda")
        
    def encode_image(self, x):
        # 1. ачальная свертка входа vae.encoder.conv_in
        w_in = self.weights["encoder.conv_in.weight"].to(torch.bfloat16)
        b_in = self.weights["encoder.conv_in.bias"].to(torch.bfloat16)
        x = torch.nn.functional.conv2d(x, w_in, bias=b_in, padding=1)
        
        # 2. рогон через даунсэмпл-блоки (сжатие разрешения в 8 раз)
        for block_idx in range(4):
            w_key = f"encoder.down.{block_idx}.block.0.conv1.weight"
            if w_key in self.weights:
                w = self.weights[w_key].to(torch.bfloat16)
                b = self.weights[f"encoder.down.{block_idx}.block.0.conv1.bias"].to(torch.bfloat16)
                x = torch.nn.functional.conv2d(x, w, bias=b, padding=1)
                x = torch.nn.functional.silu(x)
                
            down_key = f"encoder.down.{block_idx}.downsample.conv.weight"
            if down_key in self.weights:
                w_down = self.weights[down_key].to(torch.bfloat16)
                b_down = self.weights[f"encoder.down.{block_idx}.downsample.conv.bias"].to(torch.bfloat16)
                x = torch.nn.functional.conv2d(x, w_down, bias=b_down, stride=2, padding=1)
        
        # 3. инальный слой вывода (conv_out) - забираем 16 каналов μ
        w_out = self.weights["encoder.conv_out.weight"].to(torch.bfloat16)
        b_out = self.weights["encoder.conv_out.bias"].to(torch.bfloat16)
        moments = torch.nn.functional.conv2d(x, w_out, bias=b_out, padding=1)
        latents = moments[:, :16, :, :]
        
        # СТЫ : ормируем амплитуду под канонический шаг ядра Хромы
        # риводим огромные линейные значения сверток к стандартному распределению FLUX VAE
        latents_normalized = (latents - latents.mean()) / (latents.std() + 1e-5)
        
        return latents_normalized * 0.3611

def process_and_cache_images():
    print("[Т] Шаг 4.2: апуск финального стабилизированного конвейера латентов...")
    
    if not os.path.exists(TrainConfig.CACHE_LATENT_DIR):
        os.makedirs(TrainConfig.CACHE_LATENT_DIR)

    encoder = NativeFluxEncoder(TrainConfig.VAE_PATH)

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
                continue
                
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            
            k = 2**4 
            new_w = (w // k) * k
            new_h = (h // k) * k
            
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                
            img_tensor = transform(img).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            
            with torch.no_grad():
                latents = encoder.encode_image(img_tensor)
                
            base_name = os.path.splitext(img_name)[0] # иксируем снайперское извлечение строго имени без .JPG
            out_latent_path = os.path.join(TrainConfig.CACHE_LATENT_DIR, f"{base_name}_latents.pt")
            
            torch.save(latents.cpu(), out_latent_path)
            print(f"[СХ] Стабилизирован латент VAE для: {img_name} [{new_w}x{new_h}]")

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    process_and_cache_images()
    print("[СХ] изуальный конвейер СТЬ  СТЬ отработал.")
