# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from torch.utils.data import DataLoader
# from torch.optim import AdamW`nfrom transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import load_file, save_file
from PIL import Image
from diffusers import AutoencoderKL

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dataset import ChromaDataset
from src.losses import FlowMatchingLoss
from src.model_utils import inject_chroma_lora

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class ChromaDoubleBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.img_attn = nn.Module()
        self.img_attn.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.img_attn.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
    def forward(self, x, x_norm):
        qkv_out = self.img_attn.qkv(x_norm)
        proj_out = self.img_attn.proj(qkv_out[..., :3072])
        return x + proj_out

class ChromaSingleBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, 21504, bias=True)
        self.linear2 = nn.Linear(15360, hidden_dim, bias=True)
        
    def forward(self, x, x_norm):
        linear1_out = self.linear1(x_norm)[..., :3072]
        return x + linear1_out

def run_latent_heavy_training():
    print('🔥 С СТЯЩ   CHROMA1-HD (Ы Ш )...')
    
    checkpoint_path = r'Z:\AiModels\models\checkpoints\chroma1\Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors'
#    vae_path = r'Z:\AiModels\models\vae\ae.safetensors'
    jsonl_path = r'Z:\flowch\metadata.jsonl'
    cache_pt_path = r'Z:\flowch\dataset\text_cache\DSC_0465.pt'
    output_lora_path = r'Z:\flowch\chroma1_mangala_lora_latent_heavy.safetensors'
    val_dir = r'Z:\flowch\validation_latent_heavy'
    
    os.makedirs(val_dir, exist_ok=True)
    epochs = 20  
    batch_size = 1  
    lr = 2e-5  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'💻 елевое устройство: {device.upper()}')

    # агрузка данных
    dataset = ChromaDataset(jsonl_path=jsonl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#    print('📥 агрузка локального VAE для генератора слепков...')
#    vae = AutoencoderKL(
#        in_channels=3, out_channels=3,
#        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
#        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
#        block_out_channels=list((128, 256, 512, 512)), layers_per_block=2, latent_channels=16, norm_num_groups=32
#    )
#    vae.load_state_dict(load_file(vae_path), strict=False)


 print("📂 Загрузка каноничного VAE из локального SDXL чекпоинта...")
    from diffusers import AutoencoderKL
    
    # Берем VAE напрямую из проверенной рабочей модели, без интернета
    vae_path = r"Z:\AiModels\models\checkpoints\sdxl\dreamshaperXL_lightningDPMSDE_FP16.safetensors"
    vae = AutoencoderKL.from_single_file(
        vae_path,
        torch_dtype=torch.float32  # Оставляем в float32, чтобы не ловить краши декодера
    ).to("cuda")
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae.eval()

    # Собираем скелет графа на CPU
    class LatentChromaTransformerGraph(nn.Module):
        def __init__(self, patch_size=2, in_channels=16, hidden_dim=3072):
            super().__init__()
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.hidden_dim = hidden_dim
            
            self.x_embed = nn.Linear(patch_size * patch_size * in_channels, hidden_dim)
            self.norms_double = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(19)])
            self.norms_single = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(38)])
            
            self.double_blocks = nn.ModuleList([ChromaDoubleBlock(hidden_dim) for _ in range(19)])
            self.single_blocks = nn.ModuleList([ChromaSingleBlock(hidden_dim) for _ in range(38)])
            
            self.x_out = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)
            self.norm_out = nn.LayerNorm(hidden_dim)
            
        def forward(self, x_t, t, text_embeddings):
            B, C, H, W = x_t.shape 
            p = self.patch_size
            
            x_patches = x_t.unfold(2, p, p).unfold(3, p, p)
            _, _, H_p, W_p, _, _ = x_patches.shape
            
            x_patches = x_patches.permute(0, 2, 3, 4, 5, 1).contiguous().clone()
            x_flat_patches = x_patches.view(B, H_p * W_p, p * p * C)
            
            x_flat = self.x_embed(x_flat_patches)
            
            if isinstance(text_embeddings, dict):
                t5_feat = text_embeddings['t5_hidden'].mean(dim=1)[:, :self.hidden_dim].unsqueeze(1)
            else:
                t5_feat = text_embeddings.unsqueeze(1)
                
            x_flat = x_flat + t5_feat.to(x_flat.dtype)
            
            for idx, block in enumerate(self.double_blocks):
                x_norm = self.norms_double[idx](x_flat)
                x_flat = block(x_flat, x_norm)
                
            for idx, block in enumerate(self.single_blocks):
                x_norm = self.norms_single[idx](x_flat)
                x_flat = block(x_flat, x_norm)
                
            x_flat_norm = self.norm_out(x_flat)
            out_patches = self.x_out(x_flat_norm)
            out_patches = out_patches.view(B, H_p, W_p, p, p, C)
            out_latent = out_patches.permute(0, 5, 1, 3, 2, 4).contiguous()
            
            return out_latent.view(B, C, H, W)

    base_model = LatentChromaTransformerGraph()

    # 🔥 СТТЫ Ш 1: ливаем реальные веса в чистый CPU-граф  инжекции LoRA
    print('📂 Синхронизация и загрузка реальных FP8 весов трансформера Chroma1-HD...')
    checkpoint_sd = load_file(checkpoint_path)
    model_sd = base_model.state_dict()
    transferred_count = 0
    
    for k, v in checkpoint_sd.items():
        if k in model_sd:
            model_sd[k] = v.to(torch.bfloat16)
            transferred_count += 1
            
    base_model.load_state_dict(model_sd, strict=False)
    print(f'  [+] СШ Т {transferred_count} СТЯЩХ Т  Т!')

    # 🔥 СТТЫ Ш 2: ереносим реальную базовую модель в VRAM
    base_model = base_model.to(device=device, dtype=torch.bfloat16)

    # 🔥 СТТЫ Ш 3: асаживаем LoRA поверх Ь фундамента весов
    base_model = inject_chroma_lora(base_model, target_rank=16, target_alpha=32)
    base_model.train()

    # Собираем только LoRA веса для оптимизатора
    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    # optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)`n    num_training_steps = num_epochs * len(dataset)`n    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * 0.05), num_training_steps=num_training_steps)
    num_epochs = 150
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
    # Считаем шаги исходя из реального размера датасета
    num_training_steps = num_epochs * len(dataset)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(num_training_steps * 0.05), 
        num_training_steps=num_training_steps
    )
    
    criterion = FlowMatchingLoss()

    print('🎲 аморозка фиксированного латентного шума x_0 для валидации...')
    val_x_0 = torch.randn(1, 16, 128, 128, device=device, dtype=torch.bfloat16) 
    val_text_cache = torch.load(cache_pt_path, map_location=device, weights_only=True)
    val_t5_hidden = val_text_cache['t5_hidden'].mean(dim=1)[:, :3072]

    print(f'🏋️ оличество активных LoRA модулей в каноничном графе: {len(trainable_params)}')
    print('🚀 оевая печь запущена с ЬЫ первым шагом инициализации!')

    for epoch in range(epochs):
        base_model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            latent_values = batch['latent_values'].to(device, dtype=torch.bfloat16)
            text_embeddings = {
                'clip_hidden': batch['clip_hidden'].to(device),
                't5_hidden': batch['t5_hidden'].to(device)
            }
            
            loss = criterion(base_model, latent_values, text_embeddings)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()
            scheduler.step()  # <--- Вставляем сюда!
            epoch_loss += loss.item()
            
        print(f'📊 поха [{epoch+1}/{epochs}] | еальный латентный лосс: {epoch_loss / len(dataloader):.6f}')

        print(f'📸 [изуализатор] екодирование слепка epoch_{epoch+1} через VAE...')
        base_model.eval()
        with torch.no_grad():
            x_t = val_x_0.clone()
            val_steps = 20
            dt = 1.0 / val_steps
            
            for i in range(val_steps):
                v = base_model(x_t, i * dt, val_t5_hidden)
                x_t = x_t + v * dt
                
            decoded_latents = x_t / 0.18215
            pixels_out = vae.decode(decoded_latents).sample
            
            pixels_out = (pixels_out + 1.0) / 2.0
            pixels_out = pixels_out.clamp(0.0, 1.0).squeeze(0).cpu().float()
            img_array = (pixels_out.permute(1, 2, 0).numpy() * 255).astype('uint8')
            
            val_img_path = os.path.join(val_dir, f'latent_heavy_epoch_{epoch+1}.png')
            Image.fromarray(img_array).save(val_img_path)
            print(f'  [+] еткий RGB-слепок успешно испечен: {val_img_path}')

    print('💾 кстракция финальных LoRA матриц...')
    lora_state_dict = {}
    for name, param in base_model.named_parameters():
        if 'lora_' in name:
            clean_name = name.replace('original_layer.', '')
            lora_state_dict[clean_name] = param.cpu().detach()

    save_file(lora_state_dict, output_lora_path)
    print(f'🎉 Ь   Ш!')

if __name__ == '__main__':
    run_latent_heavy_training()
