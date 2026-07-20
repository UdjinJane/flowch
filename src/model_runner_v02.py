# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ РАННЕРА ===
# [Этот блок настраивает окружение и сигнатуру маршевого шага Flux-LoRA]
# [Полностью автономный контур без внешних относительных импортов]

import torch

def run_lora_model_step(
    lora_model, 
    batch, 
    packed_noisy_latents, 
    timesteps_attr, 
    prompt_embeds, 
    pooled_projections, 
    txt_ids, 
    img_ids
):
    device = torch.device("cuda")
    base_dtype = torch.float8_e4m3fn  # Целевой тип квантованного ядра Flux
    meta_dtype = torch.bfloat16       # Сигнальный тип для эмбеддеров и LoRA

# === КОНЕЦ БЛОКА 1 ===
# === БЛОК 2.1: СБОРКА И ПЕРЕХВАТ МЕТОДОВ FORWARD ===
# Динамически собираем оригинальные методы и готовим списки для отката

base_transformer = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
orig_blocks = []

# 1. Сборка двойных блоков трансформера Flux
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

# === КОНЕЦ БЛОКА 2.1 ===

# === БЛОК 2.2: ПЕРЕХВАТ ОДНОЧНОЧНЫХ БЛОКОВ ===
# Завершаем изоляцию гибридного контура кастинга для одиночных слоев Flux

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

# === КОНЕЦ БЛОКА 2.2 ===
# === БЛОК 2.3: МАРШЕВЫЙ ВЫЗОВ И ВОССТАНОВЛЕНИЕ КОНТУРА ===
# [Этот блок выполняет запуск ядра и гарантирует откат forward-методов]

try:
    # Инициализация и вызов LoRA модели (примерная логика)
    model_output = lora_model(...)
    
    # Обработка выхода (очистка размерностей)
    pred_tensor = process_output(model_output)
finally:
    # Восстановление оригинальных forward-методов для предотвращения утечек
    for block, orig_fwd in orig_blocks:
        block.forward = orig_fwd

return pred_tensor
# === КОНЕЦ ФАЙЛА MODEL_RUNNER_V02 ===
