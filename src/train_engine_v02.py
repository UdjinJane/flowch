import os
import sys
import gc
import shutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from config import TrainConfig
from generate_v02 import run_inference_v02
from dataset_v02 import get_dataloader_v02
from lora_core_v02 import FluxLoraCoreV02
from model_runner_v02 import run_lora_model_step

# --- АВТОМАТИЗИРОВАННАЯ ЗАЩИТА ОТ КРИВЫХ РУК (ИНЖЕКЦИЯ ЗОЛОТА V02) ---
try:
    from ao_optim_monolith_v02 import AdamW8bit
    USING_8BIT_OPTIM = True
except Exception as e:
    print(f"[ВНИМАНИЕ] Ошибка инжекции 8-bit монолита: {e}")
    print("[ОТК] Аварийный протокол: переключаюсь на стандартный float32 AdamW.")
    USING_8BIT_OPTIM = False

def pack_latents_to_patches(latents):
    b, c, h, w = latents.shape
    assert h % 2 == 0 and w % 2 == 0, f"Разрешение должно быть кратно 2, получили {h}x{w}"
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3)
    return latents.flatten(1, 2)

def generate_flux_img_ids(height, width, device):
    h_patches, w_patches = height // 2, width // 2
    img_ids = torch.zeros(h_patches, w_patches, 3, device=device)
    img_ids[..., 1] = torch.arange(h_patches, device=device)[:, None]
    img_ids[..., 2] = torch.arange(w_patches, device=device)[None, :]
    return img_ids.view(-1, 3)

def main_train_loop():
    print("[Т] Магистральный запуск ядра обучения: train_engine_v02")
    
    shutil.rmtree("__pycache__", ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda")
    
    print("[Т] Запуск загрузчика кэшированных эмбеддингов...")
    dataloader = get_dataloader_v02()
    
    print("[Т] Прогрев и инжекция LoRA адаптеров...")
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    trainable_params = []
    for name, param in lora_model.named_parameters():
        if "lora_" in name and any(t in name for t in TrainConfig.TARGET_MODULES):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    print(f"[УСПЕХ] Зафиксировано обучаемых тензоров: {len(trainable_params)}")
    
    # Автоматический селектор оптимизатора на основе защиты контура
    if USING_8BIT_OPTIM:
        print("[УСПЕХ] Реактор успешно переведен на экономное int8-топливо (AdamW8bit V02).")
        optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=0.01)
    
    os.makedirs(TrainConfig.LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(TrainConfig.LOGS_DIR, "train_logs.txt")
    global_step = 0
    
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        lora_model.train()
        torch.cuda.manual_seed_all(42 + epoch)
        
        for step, batch in enumerate(dataloader):
            # --- ВРЕМЕННЫЙ ОТЛАДОЧНЫЙ ЗОНД ОМНИССИИ ---
            # print(f"[ПРИБОР] Форма latents из кэша: {batch['latents'].shape}")
            # print(f"[ПРИБОР] Форма prompt_embeds из кэша: {batch['prompt_embeds'].shape}")
            # sys.exit(0) # Аварийный стоп после первого замера!
            # ------------------------------------------

            global_step += 1
            
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            b_size = latents.shape[0]
            
            noise = torch.randn_like(latents)
            t_attr = torch.rand(b_size, device=device, dtype=torch.bfloat16)
            
            packed_noisy_latents = pack_latents_to_patches((1.0 - t_attr.view(-1, 1, 1, 1)) * latents + t_attr.view(-1, 1, 1, 1) * noise)
            
            # --- ФИНАЛЬНЫЙ МАТЕМАТИЧЕСКИЙ РОПЕ-КОНТУР V04 (СТРОКИ 94-95) ---
            # Извлекаем латентную геометрию: 64 и 64
            latents_h, latents_w = latents.shape[2], latents.shape[3]
            
            # Передаем чистые латенты. Функция сама сделает // 2 и выдаст сетку 32x32 (1024 патча)
            img_ids = generate_flux_img_ids(latents_h, latents_w, device).to(torch.bfloat16)
            
            # txt_ids: строго 2D под длину текстового контекста prompt_embeds [256, 3]
            txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=torch.bfloat16)
            # --------------------------------------------------------------
            
            # --- СНАЙПЕРСКИЙ ВЫЗОВ РАННЕРА V02 (СТРОКИ 94-98) ---
            pred_tensor = run_lora_model_step(
                lora_model=lora_model, 
                batch=batch, 
                packed_noisy_latents=packed_noisy_latents,
                timesteps_attr=t_attr, 
                prompt_embeds=prompt_embeds,
                pooled_projections=torch.zeros(b_size, 768, device=device, dtype=torch.bfloat16),
                txt_ids=txt_ids, 
                img_ids=img_ids
            ) # Закрывающая скобка строго здесь!
            # ----------------------------------------------------

            
            pred_tensor = pred_tensor.to(dtype=torch.bfloat16)
            target_flow = (noise - latents).to(dtype=torch.bfloat16)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            if pred_tensor.shape != packed_target_flow.shape:
                pred_tensor = pred_tensor[:, :packed_target_flow.shape, :]
            
            loss = F.mse_loss(pred_tensor, packed_target_flow, reduction="mean")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[КРИТ] Обнаружен взрыв градиентов на шаге {global_step}!")
                sys.exit(1)
                
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            torch.cuda.empty_cache()
            
            if global_step % 10 == 0:
                current_loss = loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                log_msg = f"[ОТК] Шаг: {global_step} | Эпоха: {epoch} | MSE Лосс: {current_loss:.4f}\n"
                print(log_msg.strip())
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    lf.write(log_msg)
                    
            if global_step % TrainConfig.SAVE_STEPS == 0:
                print(f"[Т] Рубеж сохранения. Запекаем чекпоинт на шаге {global_step}...")
                checkpoint_path = os.path.join(TrainConfig.OUTPUT_DIR, f"flux_lora_step_{global_step}.safetensors")
                lora_state_dict = {k: v for k, v in lora_model.state_dict().items() if "lora_" in k}
                torch.save(lora_state_dict, checkpoint_path)
                
                lora_model.eval()
                with torch.no_grad():
                    run_inference_v02(global_step)
                lora_model.train()

    print("[УСПЕХ] Реактор завершил плавку всех эпох. Контур чист!")

if __name__ == "__main__":
    main_train_loop()
