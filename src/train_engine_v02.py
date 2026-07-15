def main_train_loop():
    print("[Т] Запуск финального экономного диспетчера: train_engine_v02")
    
    # Жесткое выжигание кэша импортов перед стартом плавки по умолчанию
    import shutil
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "__pycache__"), ignore_errors=True)
    print("[ОТК] Локальный кэш __pycache__ принудительно зачищен.")


import os
import torch
import torch.nn.functional as F
import gc
from torch.optim import AdamW
from config import TrainConfig
from dataset_v02 import get_dataloader_v02
from flow_math_v01 import FluxFlowMathV01
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

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
    print("[Т] Запуск финального экономного диспетчера: train_engine_v02")
    
    # Инициализация стерильных лоудеров и инжекция LoRA в bfloat16-трансформер
    dataloader = get_dataloader_v02()
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # Собираем только обучаемые параметры LoRA-адаптеров для оптимизатора
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE)
    
    device = torch.device("cuda")
    global_step = 0
    current_step_real = 0
    
    print(f"[Т] Реактор обкатки V02 запущен на {TrainConfig.RESOLUTION}px. Цель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    print("[Т] Начало эпохи плавки 1")
    
    lora_model.train()
    
    for batch in dataloader:
        # Извлекаем предварительно рассчитанные латенты и эмбеддинги текста из SSD-кэша с изоляцией графа
        with torch.no_grad():
            model_latents = batch["latents"].to(device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.bfloat16)
            b, c, h, w = model_latents.shape
            packed_latents = pack_latents_to_patches(model_latents)

            # Генерация маршевого шума Rectified Flow
            noise = torch.randn_like(packed_latents, device=device, dtype=torch.bfloat16)

        
        # Математика кастомного квадратичного распределения таймстепов по перфокарте
        t = torch.rand(b, device=device, dtype=torch.bfloat16)
        t = 1.0 - (t * t)


        
        # Линейный блендинг Rectified Flow шума и латентов
        t_bc = t.view(-1, 1)
        packed_noisy_latents = (1.0 - t_bc) * packed_latents + t_bc * noise
        packed_target_flow = noise - packed_latents
        
        # Формируем служебные ID векторов геометрии кадра (строго 2D для diffusers)
        img_ids_cleaned = generate_flux_img_ids(h, w, device=device)
        txt_len = int(prompt_embeds.shape[1])
        txt_ids_cleaned = torch.zeros(txt_len, 3, device=device, dtype=torch.bfloat16)
        
        timesteps_attr = t.squeeze().view(-1) * 1000.0
        pooled_projections = torch.zeros(b, 768, device=device, dtype=torch.bfloat16)
        
        # --- НАЧАЛО БЛОКА: МАРШЕВЫЙ ЗАПУСК ИЗОЛИРОВАННОГО РАННЕРА С МАСКИРОВАНИЕМ Т5 ---
        model_output = run_lora_model_step(
            lora_model, batch, packed_noisy_latents, timesteps_attr, 
            prompt_embeds, pooled_projections, txt_ids_cleaned, img_ids_cleaned
        )
        # --- КОНЕЦ БЛОКА: МАРШЕВЫЙ ЗАПУСК ИЗОЛИРОВАННОГО РАННЕРА С МАСКИРОВАНИЕМ Т5 ---
        
        pred_tensor = model_output[0]
        pred_latents = pred_tensor[:, :, :64]
        
        # Расчет MSE-лосса
        loss = F.mse_loss(pred_latents.float(), packed_target_flow.float(), reduction="mean")
        loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        global_step += 1
        
        # Проверка окна накопления градиентов (виртуальный батч)
        if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            current_step_real = global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS
            print(f"[# {current_step_real}] Маршевый Loss V02 (512px квадрат): {loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS:.6f}")
            
            # Консервация весов каждые 200 реальных шагов
            if current_step_real % 200 == 0 or current_step_real == (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
                ckpt_name = f"mng_oks_bl_flux_lora_step_{current_step_real}.safetensors"
                ckpt_path = os.path.join(TrainConfig.OUTPUT_DIR, ckpt_name)
                print(f"[Т] Выпечка LoRA чекпоинта на SSD: {ckpt_path}")
                
                lora_state_dict = FluxLoraCoreV02.get_peft_model_state_dict(lora_model)
                clean_lora_dict = {k: v.to(torch.bfloat16) for k, v in lora_state_dict.items()}
                FluxLoraCoreV02.save_file(clean_lora_dict, ckpt_path)
                print(f"[УСПЕХ] Чекпоинт {ckpt_name} запечен успешно!")
                
        if (global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS) >= (TrainConfig.MAX_TRAIN_STEPS // TrainConfig.GRADIENT_ACCUMULATION_STEPS):
            break

    # Аварийная зачистка кэша и сбор мусора после каждого шага плавки батча
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main_train_loop()
