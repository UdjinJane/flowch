# === БЛОК 1 СТАРТ ===
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

def pack_latents_to_patches(latents):
    """
    Канонический служебный маневр FLUX.
    Перегруппировывает 2D латенты из формы [B, 16, H, W] в плоские 1D патчи [B, L, 64].
    Каналы c=16 схлопываются в блоки 2x2, выдавая c_out=64.
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    latents = latents.flatten(1, 2)
    return latents

def generate_flux_img_ids(latent_h, latent_w, device):
    """
    Генератор координатной сетки слоев (img_ids) по стандарту Diffusers FLUX.
    Строит чистый 2D тензор позиций [Num_Patches, 3] без батч-размерности.
    """
    h_len = latent_h // 2
    w_len = latent_w // 2
    grid_h = torch.arange(h_len, device=device)[:, None].repeat(1, w_len)
    grid_w = torch.arange(w_len, device=device)[None, :].repeat(h_len, 1)
    
    img_ids = torch.zeros(h_len * w_len, 3, device=device, dtype=torch.bfloat16)
    img_ids[:, 1] = grid_h.flatten()
    img_ids[:, 2] = grid_w.flatten()
    return img_ids
# === БЛОК 1 ФИНАЛ ===

# === БЛОК 2 СТАРТ ===
def main_train_loop():
    print("[ОБТ] Шаг 5.5: Запуск обновленного диспетчера FLUX-Patch: train_engine_v01")
    
    if not os.path.exists(TrainConfig.OUTPUT_DIR):
        os.makedirs(TrainConfig.OUTPUT_DIR)
        
    # Подаем кэшированные тензоры с SSD
    dataloader = get_dataloader_v01()
    
    # Поднимаем 48 слоев трансформера Хромы на GPU
    lora_model = FluxLoraCoreV01.init_transformer_with_lora()
    
    # Разворачиваем оптимизатор строго над обучаемыми параметрами LoRA (18.6М)
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=1e-2)
    
    print(f"[ОБТ] Реактор обкатки запущен на {TrainConfig.RESOLUTION}px. Цель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    
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
            
            # Расчет нелинейного времени t и блендинг траектории шума Flow Matching
            t = FluxFlowMathV01.generate_train_timesteps(b, device="cuda")
            noisy_latents, target_flow = FluxFlowMathV01.blend_noise_and_latents(latents, noise, t)
            
            # Упаковка 2D латентов и шума в плоские патчи 2x2 под требования FLUX
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Подготовка временных атрибутов и пространственных координат патчей
            img_ids = generate_flux_img_ids(h, w, device="cuda")
            timesteps_attr = t.squeeze().view(-1) * 1000.0
            
            # СНАЙПЕРСКИЙ ФИКС: Извлекаем строго количество токенов (256) как одномерное целое число!
            txt_len = prompt_embeds.shape[0]
            txt_ids = torch.zeros(txt_len, 3, device="cuda", dtype=torch.bfloat16)
            
            # Генерируем пустой, валидный pooled_projections формы [B, 768] для обхода None-краша
            pooled_projections = torch.zeros(b, 768, device="cuda", dtype=torch.bfloat16)
# === БЛОК 2 ФИНАЛ ===

# === БЛОК 3 СТАРТ ===
            # ПРЯМОЙ ШАГ ИНФЕРЕНСА ЧЕРЕЗ ТРАНСФОРМЕР ХРОМЫ (23.7 ГБ VRAM)
            model_output = lora_model(
                hidden_states=packed_noisy_latents,
                timestep=timesteps_attr,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False
            )
            
            # Извлекаем тензор из кортежа выдачи модели
            pred_tensor = model_output
            
            # КРИТИЧЕСКИЙ МАТЕМАТИЧЕСКИЙ ФИКС: Отрезаем первые 64 канала латентных патчей!
            # Полностью выравниваем форму под packed_target_flow [1, 1024, 64]
            pred_latents = pred_tensor[:, :, :64]
            
            # Вычисляем MSE Loss на плоских патчах скорости изменения шума Flow Matching
            loss = F.mse_loss(pred_latents.float(), packed_target_flow.float(), reduction="mean")
            
            # Накопление градиентов под малый размер батча
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            global_step += 1
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_step_real = global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS
                print(f"[ШАГ {current_step_real}] Лосс траектории шума (512px): {loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS:.6f}")
                
                # ИНЖЕНЕРНАЯ ВЫПЕЧКА ЧЕКПОИНТОВ LORA НА SSD КАЖДЫЕ 200 ШАГОВ
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

    print("[УСПЕХ] Обкатка реактора на 512px завершена безупречно!")

if __name__ == "__main__":
    import shutil
    # Физически сжигаем кэш компиляции Python перед каждым стартом
    if os.path.exists("src/__pycache__"):
        shutil.rmtree("src/__pycache__")
        
    main_train_loop()
# === БЛОК 3 ФИНАЛ ===
