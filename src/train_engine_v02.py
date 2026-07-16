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

# Жесткое выжигание кэша
shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__"), ignore_errors=True)
print("[ОТК] Локальный кэш __pycache__ принудительно зачищен.")


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
    
    # Инициализация лоудеров и инжекция LoRA с принудительной очисткой памяти
    import gc
    dataloader = get_dataloader_v02()
    # Жестко переводим лоудер в однопоточный режим во избежание deadlock в Windows
    dataloader.num_workers = 0
    
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    # Выжигаем остаточный мусор из ОЗУ и VRAM перед пуском
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- ДАТЧИКИ ТОТАЛЬНОГО ФИЗИЧЕСКОГО КОНТРОЛЯ ГРАДИЕНТОВ LORA START ---
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    print(f"[ОТК] >>> ФИЗИЧЕСКИЙ КОНТРОЛЬ ЯДРА LoRA <<<")
    print(f" └── Найдено обучаемых тензоров в ОЗУ: {len(trainable_params)}")
    total_trainable_elements = sum(p.numel() for p in trainable_params)
    print(f" └── Общее число обучаемых весов: {total_trainable_elements}")
    if len(trainable_params) == 0:
        print(" [КРИТИЧЕСКИЙ ОТКАЗ] ЛОРА АДАПТЕР ПОЛНОСТЬЮ ОБЕЗГЛАВЛЕН! Градиенты заблокированы!")
    # --- ДАТЧИКИ ТОТАЛЬНОГО ФИЗИЧЕСКОГО КОНТРОЛЯ ГРАДИЕНТОВ LORA END ---

    # Собираем только обучаемые параметры LoRA-адаптеров для оптимизатора
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE)
    device = torch.device("cuda")
    global_step = 0
    current_step_real = 0
    
    print(f"[Т] Реактор обкатки V02 запущен на {TrainConfig.RESOLUTION}px. Цель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    print("[Т] Начало эпохи плавки 1")
    
    lora_model.train()
    
    # Датчики времени для прогнозирования трудозатрат
    import time
    start_time = time.time()
    step_timestamps = []
    
    epoch = 1
    run_reactor = True
    
    print(f"[Т] Главные маршевые двигатели запущены. Ожидание стабилизации тяги...")
    
    while run_reactor:
        print(f"[Т] Вход в эпоху плавки № {epoch}")
        
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
            
            # Кэшируем физическое среднее таймстепа для анализа динамики лосса
            avg_t = t.mean().item()
            
            # Линейный блендинг Rectified Flow шума и латентов
            t_bc = t.view(-1, 1)
            packed_noisy_latents = (1.0 - t_bc) * packed_latents + t_bc * noise
            packed_target_flow = noise - packed_latents
            
            # Формируем служебные ID векторов геометрии кадра (строго 2D для diffusers)
            img_ids_cleaned = generate_flux_img_ids(h, w, device=device)
            txt_len = int(prompt_embeds.shape[1])
            txt_ids_cleaned = torch.zeros(txt_len, 3, device=device, dtype=torch.bfloat16)
            
            # Передаем чистый физический таймстеп [B] для Rectified Flow весов ComfyUI без умножения на 1000
            timesteps_attr = t.view(-1)
            pooled_projections = torch.zeros(b, 768, device=device, dtype=torch.bfloat16)
            
            # Маршевый запуск изолированного раннера с маскированием Т5
            model_output = run_lora_model_step(
                lora_model, batch, packed_noisy_latents, timesteps_attr, 
                prompt_embeds, pooled_projections, txt_ids_cleaned, img_ids_cleaned
            )
            
            # Точное извлечение тензора: если кортеж, берем нулевой элемент
            pred_tensor = model_output[0] if isinstance(model_output, tuple) else model_output.sample
            pred_latents = pred_tensor[:, :, :64]

            
            # Расчет MSE-лосса
            loss = F.mse_loss(pred_latents.float(), packed_target_flow.float(), reduction="mean")
            loss = loss / TrainConfig.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Физический замер протечки градиентов в матрицы LoRA
            # [ОТК] Бортовой термометр: мониторинг динамики градиентов каждые 10 шагов плавки
            if current_step_real % 10 == 0 or current_step_real == 0:
                grads = [p.grad.abs().mean().item() for p in trainable_params if p.grad is not None]
                avg_grad = sum(grads) / len(grads) if grads else 0.0
                print(f"\n[ОТК] >>> ТЕЛЕМЕТРИЯ РЕАКТОРА (Шаг #{current_step_real}) <<<")
                print(f" └── Живых тензоров LoRA под нагрузкой: {len(grads)} из {len(trainable_params)}")
                print(f" └── Актуальная амплитуда дельты градиента: {avg_grad:.8f}")
                if avg_grad == 0.0:
                    print(" [КРИТИЧЕСКИЙ ОТКАЗ] ПОТОК ГРАДИЕНТОВ ПРЕРВАН НАМЕРТВО!")

            
            global_step += 1
            
            # Проверка окна накопления градиентов (виртуальный батч)
            if global_step % TrainConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_step_real = global_step // TrainConfig.GRADIENT_ACCUMULATION_STEPS
                
                # Расчет скорости и ETA
                step_timestamps.append(time.time())
                if len(step_timestamps) > 1:
                    step_time = step_timestamps[-1] - step_timestamps[-2]
                else:
                    step_time = time.time() - start_time
                
                steps_left = TrainConfig.MAX_TRAIN_STEPS - current_step_real
                eta_seconds = steps_left * step_time
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                
                # Лог запекания: [# Шаг] Loss | Скорость | Осталось | Физика шума (Средний t)
                print(f"[# {current_step_real}/{TrainConfig.MAX_TRAIN_STEPS}] "
                      f"Loss: {loss.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS:.5f} | "
                      f"T-Step: {step_time:.2f}s | ETA: {eta_str} | "
                      f"Физика: avg_t={avg_t:.3f} (эпоха {epoch})")
                
                # Консервация весов каждые 10 реальных шагов - 1 эпоха.
                # Генерация сэмпла и лоры каждую эпоху для контроля динамики предмета
                if current_step_real % 10 == 0 or current_step_real == TrainConfig.MAX_TRAIN_STEPS:

                # Ручной запуск визуализации текущей эпохи плавки перед выпечкой весов
                    run_inference_v02(
                        loaded_transformer=lora_model,
                        current_step=current_step_real,
                        text_embedding=prompt_embeds,
                        steps=25,
                        device=device
                    )

                    ckpt_name = f"mng_oks_bl_flux_lora_step_{current_step_real}.safetensors"
                    ckpt_path = os.path.join(TrainConfig.OUTPUT_DIR, ckpt_name)
                    print(f"[Т] Выпечка LoRA чекпоинта на SSD: {ckpt_path}")
                    
                    lora_state_dict = FluxLoraCoreV02.get_peft_model_state_dict(lora_model)
                    clean_lora_dict = {k: v.to(torch.bfloat16) for k, v in lora_state_dict.items()}
                    FluxLoraCoreV02.save_file(clean_lora_dict, ckpt_path)
                    print(f"[УСПЕХ] Чекпоинт {ckpt_name} запечен успешно!")
                
                if current_step_real >= TrainConfig.MAX_TRAIN_STEPS:
                    run_reactor = False
                    break
        
        epoch += 1

if __name__ == "__main__":
    main_train_loop()
