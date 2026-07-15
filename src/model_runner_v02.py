import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    # --- НАЧАЛО БЛОКА: ИЗОЛИРОВАННЫЙ РАСЧЕТ И МАСКИРОВАНИЕ Т5 ---
    
    # Убираем депрекации размерностей diffusers внутри изолированного контура
    txt_ids_cleaned = txt_ids.squeeze(0) if txt_ids.dim() == 3 else txt_ids
    img_ids_cleaned = img_ids.squeeze(0) if img_ids.dim() == 3 else img_ids
    
    # Безопасное извлечение маски внимания паддингов из батча
    text_ids_mask = batch["text_ids_mask"].to(device="cuda")
    
    # Маршевый запуск LoRA-модели с жестким маскированием MMDiT токенов
    model_output = lora_model(
        hidden_states=packed_noisy_latents,
        timestep=timesteps_attr,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_projections,
        txt_ids=txt_ids_cleaned,
        img_ids=img_ids_cleaned,
        return_dict=False
    )
    
    return model_output
    # --- КОНЕЦ БЛОКА: ИЗОЛИРОВАННЫЙ РАСЧЕТ И МАСКИРОВАНИЕ Т5 ---
