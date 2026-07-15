import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from config import TrainConfig
from dataset_v01 import get_dataloader_v01
from flow_math_v01 import FluxFlowMathV01
from lora_core_v01 import FluxLoraCoreV01

# С СЫ  FLUX: паковка латентов в плоские патчи 2x2
def pack_latents_to_patches(latents):
    b, c, h, w = latents.shape
    # ерегруппировываем пиксели в блоки 2x2: c=16 превращается в c_out=64
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2) # Схлопываем высоту и ширину в один вектор токенов
    return latents

# Т Т СТ С (IMG_IDS)  СТТ DIFFUSERS FLUX
def generate_flux_img_ids(batch_size, latent_h, latent_w, device):
    # Сетка строится для патчей 2x2, поэтому размеры делим на 2
    h_len = latent_h // 2
    w_len = latent_w // 2
    
    grid_h = torch.arange(h_len, device=device)[:, None].repeat(1, w_len)
    grid_w = torch.arange(w_len, device=device)[None, :].repeat(h_len, 1)
    
    # Собираем 3D координатный вектор позиций
    img_ids = torch.zeros(h_len * w_len, 3, device=device, dtype=torch.bfloat16)
    img_ids[:, 1] = grid_h.flatten()
    img_ids[:, 2] = grid_w.flatten()
    
    return img_ids.unsqueeze(0).repeat(batch_size, 1, 1)

def main_train_loop():
    print("[Т] Шаг 5.5: апуск обновленного диспетчера FLUX-Patch: train_engine_v01")
    
    if not os.path.exists(TrainConfig.OUTPUT_DIR):
        os.makedirs(TrainConfig.OUTPUT_DIR)
        
    dataloader = get_dataloader_v01()
    lora_model = FluxLoraCoreV01.init_transformer_with_lora()
    
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=1e-2)
    
    print(f"[Т] еактор обкатки запущен на {TrainConfig.RESOLUTION}px. ель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    
    global_step = 0
    epoch = 0
    
    while global_step < TrainConfig.MAX_TRAIN_STEPS:
        epoch += 1
        print(f"[Т] ачало эпохи плавки {epoch}")
        
        for batch in dataloader:
            latents = batch["latents"].to(device="cuda", dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device="cuda", dtype=torch.bfloat16)
            text_ids_mask = batch["text_ids_mask"].to(device="cuda", dtype=torch.bfloat16)
            
            b, c, h, w = latents.shape
            noise = torch.randn_like(latents)
            
            # асчет времени и блендинг траектории
            t = FluxFlowMathV01.generate_train_timesteps(b, device="cuda")
            noisy_latents, target_flow = FluxFlowMathV01.blend_noise_and_latents(latents, noise, t)
            
            # ---  ТТ  Ш  Т  СТТ Я FLUX ---
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # енерация координатной сетки для внимания
            img_ids = generate_flux_img_ids(b, h, w, device="cuda")
            
            # ектор времени и маска
            timesteps_attr = t.squeeze().view(-1) * 1000.0
            ja_kwargs = {"text_ids_mask": text_ids_mask}
            
            # Текстовые координаты (для T5 во FLUX они всегда нулевые фиксированной длины)
            txt_ids = torch.zeros(b, prompt_embeds.shape, 3, device="cuda", dtype=torch.bfloat16)
            
            # Я Ш С  48 С ТС ХЫ
            model_pred = lora_model(
                hidden_states=packed_noisy_latents,
                timestep=timesteps_attr,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=ja_kwargs,
                return_dict=False
            )
            
            # ычисляем MSE Loss на плоских патчах скорости изменения шума
            loss = F.mse_loss(model_pred.float(), packed_target_flow.float(), reduction="mean")
            
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            global_step += 1
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_step_real = global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS
                print(f"[Ш {current_step_real}] Тестовый Loss (512px квадрат): {loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS:.6f}")
                
                if current_step_real % 200 == 0 or global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS == (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
                    ckpt_name = f"mng_oks_bl_flux_lora_step_{current_step_real}.safetensors"
                    ckpt_path = os.path.join(TrainConfig.OUTPUT_DIR, ckpt_name)
                    print(f"[Т] ыпечка LoRA чекпоинта на SSD: {ckpt_path}")
                    
                    lora_state_dict = get_peft_model_state_dict(lora_model)
                    clean_lora_dict = {k: v.to(torch.bfloat16) for k, v in lora_state_dict.items()}
                    save_file(clean_lora_dict, ckpt_path)
                    print(f"[СХ] екпоинт {ckpt_name} запечен!")
                    
            if (global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS) >= (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
                break

    print("[СХ] бкатка реактора на 512px завершена безупречно!")

if __name__ == "__main__":
    import shutil
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    main_train_loop()
