# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ РАННЕРА ===
# [Этот блок настраивает окружение и сигнатуру маршевого шага Flux-LoRA]
import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    base_dtype = torch.float8_e4m3fn  # Целевой тип квантованного ядра Flux
    meta_dtype = torch.bfloat16      # Сигнальный тип для эмбеддеров и LoRA
# === КОНЕЦ БЛОКА 1 ===

# === БЛОК 2: МОНКИ-ПАЧИНГ ПРОЦЕССОРА И МАРШЕВЫЙ ВЫЗОВ МОДЕЛИ ===

    # Извлекаем базовый трансформер и запоминаем процессор
    base_transformer = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
    orig_processor = base_transformer.processor
    
    # Кастомный шлюз для авто-кастинга типов
    def hybrid_cast_processor(attn, hidden_states, encoder_hidden_states=None, *args, **kwargs):
        if hidden_states is not None and hidden_states.dtype != base_dtype:
            hidden_states = hidden_states.to(dtype=base_dtype)
        return orig_processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)
        
    base_transformer.processor = hybrid_cast_processor

    try:
        # Маршевый вызов модели
        model_output = lora_model(
            hidden_states=packed_noisy_latents.to(device=device, dtype=meta_dtype),
            timestep=timesteps_attr.to(device=device, dtype=meta_dtype),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=meta_dtype),
            return_dict=False
        )
        
        # Распаковка и приведение геометрии
        pred_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
        if pred_tensor.dim() == 4:
            pred_tensor = pred_tensor.squeeze(1)
            
    finally:
        # Деинсталляция шлюза
        base_transformer.processor = orig_processor

    return pred_tensor.to(dtype=meta_dtype)
# === КОНЕЦ БЛОКА 2 ===
