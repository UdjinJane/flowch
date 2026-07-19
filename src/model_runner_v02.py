# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ РАННЕРА ===
# [Этот блок настраивает окружение и сигнатуру маршевого шага Flux-LoRA]
import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    base_dtype = torch.float8_e4m3fn  # Целевой тип квантованного ядра Flux
    meta_dtype = torch.bfloat16      # Сигнальный тип для эмбеддеров и LoRA
# === КОНЕЦ БЛОКА 1 ===

# === БЛОК 2: ТОЧЕЧНЫЙ ПЕРЕХВАТ БЛОКОВ ВНИМАНИЯ И МАРШЕВЫЙ ШАГ ===
# [Этот блок динамически кастит hidden_states на уровне каждого блока Flux]

    base_transformer = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model

    # Сохраняем оригинальные методы forward для двух типов блоков Flux
    orig_blocks = []
    
    # 1. Патчим двойные блоки трансформера
    if hasattr(base_transformer, "transformer_blocks"):
        for block in base_transformer.transformer_blocks:
            orig_fwd = block.forward
            orig_blocks.append((block, orig_fwd))
            
            def make_hybrid_fwd(old_fwd):
                def hybrid_fwd(hidden_states, encoder_hidden_states=None, *args, **kwargs):
                    if hidden_states is not None and hidden_states.dtype != base_dtype:
                        hidden_states = hidden_states.to(dtype=base_dtype)
                    if encoder_hidden_states is not None and encoder_hidden_states.dtype != base_dtype:
                        encoder_hidden_states = encoder_hidden_states.to(dtype=base_dtype)
                    return old_fwd(hidden_states, encoder_hidden_states, *args, **kwargs)
                return hybrid_fwd
                
            block.forward = make_hybrid_fwd(orig_fwd)

    # 2. Патчим одиночные блоки трансформера
    if hasattr(base_transformer, "single_transformer_blocks"):
        for block in base_transformer.single_transformer_blocks:
            orig_fwd = block.forward
            orig_blocks.append((block, orig_fwd))
            
            def make_hybrid_single_fwd(old_fwd):
                def hybrid_single_fwd(hidden_states, *args, **kwargs):
                    if hidden_states is not None and hidden_states.dtype != base_dtype:
                        hidden_states = hidden_states.to(dtype=base_dtype)
                    return old_fwd(hidden_states, *args, **kwargs)
                return hybrid_single_fwd
                
            block.forward = make_hybrid_single_fwd(orig_fwd)

    try:
        # Маршевый проход через изолированный раннер
        model_output = lora_model(
            hidden_states=packed_noisy_latents.to(device=device, dtype=meta_dtype),
            timestep=timesteps_attr.to(device=device, dtype=meta_dtype),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=meta_dtype),
            pooled_projections=pooled_projections.to(device=device, dtype=meta_dtype),
            txt_ids=txt_ids.to(device=device, dtype=meta_dtype),
            img_ids=img_ids.to(device=device, dtype=meta_dtype),
            return_dict=False
        )
        
        pred_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
        if pred_tensor.dim() == 4:
            pred_tensor = pred_tensor.squeeze(1)
            
    finally:
        # Восстановление заводских методов forward во избежание утечки памяти
        for block, orig_fwd in orig_blocks:
            block.forward = orig_fwd

    return pred_tensor.to(dtype=meta_dtype)
# === КОНЕЦ БЛОКА 2 ===
