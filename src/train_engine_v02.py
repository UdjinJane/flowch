# === ГЛАВНЫЙ ДИСПЕТЧЕР V02 СТАРТ ===
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from config import TrainConfig
from dataset_v02 import get_dataloader_v02
from flow_math_v01 import FluxFlowMathV01
from lora_core_v02 import FluxLoraCoreV02

def pack_latents_to_patches(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2)
    return latents

def generate_flux_img_ids(latent_h, latent_w, device):
    h_len = latent_h // 2
    w_len = latent_w // 2
    grid_h = torch.arange(h_len, device=device)[:, None].repeat(1, w_len)
    grid_w = torch.arange(w_len, device=device)[None, :].repeat(h_len, 1)
    
    img_ids = torch.zeros(h_len * w_len, 3, device=device, dtype=torch.bfloat16)
    img_ids[:, 1] = grid_h.flatten()
    img_ids[:, 2] = grid_w.flatten()
    return img_ids

def main_train_loop():
    print("[ОБТ] Шаг 5.5: Запуск финального экономного диспетчера: train_engine_v02")
    
    if not os.path.exists(TrainConfig.OUTPUT_DIR):
        os.makedirs(TrainConfig.OUTPUT_DIR)
        
    # Подгружаем стерильный датасет V02 (без варнингов в консоли)
    dataloader = get_dataloader_v02()
    
    # Поднимаем трансформер с активированным градиентным чекпоинтингом
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=1e-2)
    
    print(f"[ОБТ] Реактор обкатки V02 запущен на {TrainConfig.RESOLUTION}px. Цель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    
    global_step = 0
    epoch = 0
    
    while global_step < TrainConfig.MAX_TRAIN_STEPS:
        epoch += 1
        print(f"[ОБТ] Начало эпохи плавки №{epoch}")
        
        for batch in dataloader:
            latents = batch["latents"].to(device="cuda", dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device="cuda", dtype=torch.bfloat16)
            
            b, c, h, w = latents.shape
            noise = torch.randn_like(latents)
            
            # Расчет нелинейного времени и блендинг траектории шума
            t = FluxFlowMathV01.generate_train_timesteps(b, device="cuda")
            noisy_latents, target_flow = FluxFlowMathV01.blend_noise_and_latents(latents, noise, t)
            
            # Упаковка в патчи
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Временные атрибуты и координатные сетки патчей
            img_ids = generate_flux_img_ids(h, w, device="cuda")
            timesteps_attr = t.squeeze().view(-1) * 1000.0
            
            # Извлекаем строго количество токенов (256) как одномерное целое число
            txt_len = int(prompt_embeds.shape[0])
            txt_ids = torch.zeros(txt_len, 3, device="cuda", dtype=torch.bfloat16)
            
            # Генерируем пустой pooled_projections формы [B, 768]
            pooled_projections = torch.zeros(b, 768, device="cuda", dtype=torch.bfloat16)
            
            # ПРЯМОЙ ШАГ ИНФЕРЕНСА ЧЕРЕЗ ТРАНСФОРМЕР ХРОМЫ
            model_output = lora_model(
                hidden_states=packed_noisy_latents,
                timestep=timesteps_attr,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False
            )
            
            # Извлекаем чистый тензор из кортежа выдачи
            pred_tensor = model_output
            
            # Отрезаем первые 64 канала латентных патчей, отбрасывая контекстный хвост
            pred_latents = pred_tensor[:, :, :64]
            
            # Вычисляем MSE Loss траектории Rectified Flow
            loss = F.mse_loss(pred_latents.float(), packed_target_flow.float(), reduction="mean")
            
            # Накопление градиентов под малый размер батча
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            global_step += 1
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_step_real = global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS
                print(f"[ШАГ {current_step_real}] Маршевый Loss V02 (512px квадрат): {loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS:.6f}")
                
                # Запекание чекпоинтов на SSD каждые 200 шагов
                if current_step_real % 200 == 0 or global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS == (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
                    ckpt_name = f"mng_oks_bl_flux_lora_step_{current_step_real}.safetensors"
                    ckpt_path = os.path.join(TrainConfig.OUTPUT_DIR, ckpt_name)
                    print(f"[ОТК] Выпечка LoRA чекпоинта на SSD: {ckpt_path}")
                    
                    lora_state_dict = get_peft_model_state_dict(lora_model)
                    clean_lora_dict = {k: v.to(torch.bfloat16) for k, v in lora_state_dict.items()}
                    save_file(clean_lora_dict, ckpt_path)
                    print(f"[УСПЕХ] Чекпоинт {ckpt_name} запечен успешно!")
                    
            if (global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS) >= (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
                break

    print("[УСПЕХ] Обкатка реактора V02 на 512px завершена безупречно!")

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    main_train_loop()
# === ГЛАВНЫЙ ДИСПЕТЧЕР V02 ФИНАЛ ===
