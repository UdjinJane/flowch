import os
import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torchvision.utils import save_image

from src.config import VAE_PATH, OUTPUT_DIR, device, num_epochs
from src.models import EmptyTransformer, build_vae
from src.model_utils import inject_chroma_lora
from src.dataset import ChromaDataset
from src.losses import target_velocity_loss  # Или кастомный лосс из src.losses

def compute_flow_matching_loss(model, x_1, condition):
    # Математика траектории Rectified Flow
    x_0 = torch.randn_like(x_1)
    t = torch.rand((x_1.shape[0],), device=device)
    t_blended = t.view(-1, 1, 1, 1)
    
    # Интерполяция от шума x_0 к чистым латентам x_1
    x_t = (1.0 - t_blended) * x_0 + t_blended * x_1
    target_velocity = x_1 - x_0
    
    # Прогноз модели
    v_pred = model(x_t, t, condition)
    loss = torch.mean((v_pred - target_velocity) ** 2)
    return loss

def run_latent_heavy_training():
    print('🚀 Инициализация боевого цикла обучения...')
    vae = build_vae()
    transformer = EmptyTransformer().to(device)
    
    # Внедряем LoRA адаптеры в замороженный граф
    transformer = inject_chroma_lora(transformer)
    
    # Настраиваем оптимизатор только на LoRA веса
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    
    # Инициализация датасета (41 кадр мангала)
    dataset = ChromaDataset(jsonl_path=os.path.join(OUTPUT_DIR, 'metadata.jsonl'))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_epochs * 0.05), num_training_steps=num_epochs)
    
    print('🔥 Старт плавки. Поехали!')
    for epoch in range(1, num_epochs + 1):
        transformer.train()
        total_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Предварительно подготовленные латенты или чистые картинки
            images = batch['image'].to(device)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                
            loss = compute_flow_matching_loss(transformer, latents, condition=None)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f' Epoch [{epoch}/{num_epochs}] | Loss: {total_loss / len(dataloader):.4f}')
        
        # Валидация каждые 5 эпох
        if epoch % 5 == 0:
            print(f'💾 Сохранение промежуточного слепка на эпохе {epoch}...')
            # Логика сохранения весов на SSD...

if __name__ == '__main__':
    run_latent_heavy_training()
