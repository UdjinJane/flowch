import torch
import contextlib

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    meta_dtype = torch.bfloat16

    # 1. Принудительное восстановление весов нормализации из деструктивного FP8 в bfloat16
    if not hasattr(run_lora_model_step, "_norms_upcasted"):
        for name, module in lora_model.named_modules():
            # Захватываем любые RMSNorm, LayerNorm и модуляторы слоев Flux
            if "norm" in name.lower() or "ln" in name.lower():
                module.to(dtype=meta_dtype)
        run_lora_model_step._norms_upcasted = True

    # 2. Выравнивание таймстепа до 1D-вектора под размер батча
    if timesteps_attr is not None:
        t_vector = timesteps_attr.reshape(-1)[:packed_noisy_latents.shape[0]]
    else:
        t_vector = timesteps_attr

    # 3. Однократная телеметрия для контроля шины
    if not hasattr(run_lora_model_step, "_telemetry_fired"):
        print("\n" + "="*50)
        print("[ТЕЛЕМЕТРИЯ МОСТИКА] Вход по прямому функциональному контуру:")
        print(f" -> hidden_states: {list(packed_noisy_latents.shape)}")
        print("="*50 + "\n")
        run_lora_model_step._telemetry_fired = True


#---------- Финальная интеграция Chroma.forward()
    # 4. Нативная коммутация под сигнатуру Chroma
    target_engine = lora_model.base_model.model if hasattr(lora_model, "base_model") else lora_model
    
    # Подготовка тензоров в bfloat16
    txt_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.bfloat16)
    guidance = batch.get("guidance", torch.ones(prompt_embeds.shape[0], device=device, dtype=torch.bfloat16)).to(device, torch.bfloat16)
        
    out = target_engine(
        img=packed_noisy_latents.to(device, torch.bfloat16),
        img_ids=img_ids.to(device, torch.bfloat16),
        txt=prompt_embeds.to(device, torch.bfloat16),
        txt_ids=txt_ids.to(device, torch.bfloat16),
        txt_mask=txt_mask,
        timesteps=t_vector.to(device, torch.bfloat16) if t_vector is not None else None,
        guidance=guidance
    )
#--------- Конец интеграции

    # 4. Обработка выхода диффузионного ядра (извлекаем первый элемент из кортежа)
    pred_tensor = out[0] if isinstance(out, tuple) else out
    if pred_tensor.dim() == 4:
        pred_tensor = pred_tensor.squeeze(1)

    return pred_tensor.to(dtype=meta_dtype)


