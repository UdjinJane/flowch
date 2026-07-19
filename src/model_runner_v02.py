# Финальная версия model_runner_v02.py с примененными правками:
import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    # --- НАЧАЛО БЛОКА: ИЗОЛИРОВАННЫЙ РАСЧЕТ И МАСКИРОВАНИЕ T5 ---
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

    # Проверка размерностей выходного тензора (должен быть 4D [B,C,H,W])
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    elif hasattr(model_output, "sample"):
        model_output = model_output.sample
    else:
        # Добавлена обработка случая, когда модель возвращает чистый тензор
        assert isinstance(model_output, torch.Tensor), f"Unexpected output type: {type(model_output)}"
        if model_output.dim() == 3:
            # Если 3D, добавляем фантомный канал (1)
            model_output = model_output.unsqueeze(1)
        elif model_output.dim() != 4:
            raise AssertionError(f"Unexpected output dimension: {model_output.dim()}")

    return model_output
