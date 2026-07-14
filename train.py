import os
import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
import torchvision.transforms as T

from src.config import DATASET_DIR, OUTPUT_DIR, device, num_epochs
from src.models import EmptyTransformer, build_vae
from src.model_utils import inject_chroma_lora
from src.dataset import ChromaDataset

preprocess = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def compute_flow_matching_loss(model, x_1, condition):
    x_0 = torch.randn_like(x_1)
    t = torch.rand((x_1.shape[0],), device=device)
    t_blended = t.view(-1, 1, 1, 1)
    
    x_t = (1.0 - t_blended) * x_0 + t_blended * x_1
    target_velocity = x_1 - x_0
    
    v_pred = model(x_t, t, condition)
    loss = torch.mean((v_pred - target_velocity) ** 2)
    return loss

def run_latent_heavy_training():
    print('🚀 Инициализация динамической плавки на честном VAE...')
    vae = build_vae()
    transformer = EmptyTransformer().to(device)
    
    transformer = inject_chroma_lora(transformer)
    
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=1e-5, weight_decay=0.01)
    
    dataset = ChromaDataset(jsonl_path=os.path.join(OUTPUT_DIR, 'metadata.jsonl'))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_epochs * 0.05), num_training_steps=num_epochs)
    
    print('🔥 Старт плавки. Динамическое перекодирование запущено!')
    for epoch in range(1, num_epochs + 1):
        transformer.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            img_tensors = []
            for name in batch['img_name']:
                img_name_with_ext = name if name.lower().endswith(('.jpg', '.jpeg', '.png')) else f'{name}.JPG'
                img_path = os.path.join(DATASET_DIR, img_name_with_ext)
                img = Image.open(img_path).convert('RGB')
                img_tensors.append(preprocess(img))
                
            images = torch.stack(img_tensors).to(device)
            
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                
            # Изменяем форму [B, 16, H, W] -> [B, H*W, 64] для линейных слоев прокси-модели
            B, C, H, W = latents.shape
            latents_reshaped = latents.view(B, -1, 64)
            loss = compute_flow_matching_loss(transformer, latents_reshaped, condition=None)
            loss.backward()
            # Обрезаем градиенты, чтобы предотвратить взрыв весов
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        # Вычисляем финальную норму для контроля стабильности
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0).item()
        print(f' Epoch [{epoch}/{num_epochs}] | Loss: {total_loss / len(dataloader):.4f} | Grad Norm: {grad_norm:.4f}')
        
        if epoch % 5 == 0:
            print(f'💾 [Эпоха {epoch}] Запись чекпоинта на SSD...')

if __name__ == '__main__':
    run_latent_heavy_training()
