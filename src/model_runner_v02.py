import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    """
    Маршевый модуль запуска шага Flux-LoRA модели.
    Выполняет жесткое приведение типов данных к bfloat16 для предотвращения
    переполнения (Overflow) в слоях модуляции AdaLayerNorm-Zero и устраняет
    сбои смешанной точности (Mixed Precision Cast Failures).
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Принудительный кастинг всех тензоров в bfloat16 на GPU
    txt_ids = txt_ids.to(device=device, dtype=dtype)
    img_ids = img_ids.to(device=device, dtype=dtype)
    timesteps_attr = timesteps_attr.to(device=device, dtype=dtype)
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_projections = pooled_projections.to(device=device, dtype=dtype)
    packed_noisy_latents = packed_noisy_latents.to(device=device, dtype=dtype)

    # Запуск LoRA-модели через кастомный граф инжекции
    model_output = lora_model(
        hidden_states=packed_noisy_latents,
        timestep=timesteps_attr,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_projections,
        txt_ids=txt_ids,
        img_ids=img_ids,
        return_dict=False
    )

    # Валидация и распаковка выходного тензора скорости (flow prediction)
    if isinstance(model_output, tuple):
        pred_tensor = model_output[0]
    elif hasattr(model_output, "sample"):
        pred_tensor = model_output.sample
    else:
        pred_tensor = model_output

    # Если модель выдала 4D (например, из-за внутренних оберток), сжимаем в 3D [B, L, C]
    if pred_tensor.dim() == 4:
        pred_tensor = pred_tensor.squeeze(1)
        
    return pred_tensor
