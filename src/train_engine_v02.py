import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import TrainConfig
from dataset_v02 import get_dataloader_v02
from flow_math_v01 import FluxFlowMathV01
from lora_core_v02 import FluxLoraCoreV02

# [Функции упаковки и генерации ID]
def pack_latents_to_patches(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    return latents.flatten(1, 2)

def generate_flux_img_ids(latent_h, latent_w, device):
    h_len, w_len = latent_h // 2, latent_w // 2
    grid_h = torch.arange(h_len, device=device)[:, None].repeat(1, w_len)
    grid_w = torch.arange(w_len, device=device)[None, :].repeat(h_len, 1)
    img_ids = torch.zeros(h_len * w_len, 3, device=device, dtype=torch.bfloat16)
    img_ids[:, 1], img_ids[:, 2] = grid_h.flatten(), grid_w.flatten()
    return img_ids

def main_train_loop():
    # Инициализация и цикл обучения...
    dataloader, lora_model = get_dataloader_v02(), FluxLoraCoreV02.init_transformer_with_lora()
    optimizer = AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=TrainConfig.LEARNING_RATE)
    
    for batch in dataloader:
        # Подготовка данных, forward pass (с packed_noisy_latents, img_ids_cleaned, attention_mask=text_ids_mask)
        # backward pass и optimizer.step()
        pass

if __name__ == "__main__":
    main_train_loop()
