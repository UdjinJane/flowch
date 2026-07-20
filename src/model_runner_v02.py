# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ РАННЕРА ===
# [Этот блок настраивает окружение и сигнатуру маршевого шага Flux-LoRA]
import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    base_dtype = torch.float8_e4m3fn  # Целевой тип квантованного ядра Flux
    meta_dtype = torch.bfloat16      # Сигнальный тип для эмбеддеров и LoRA
# === КОНЕЦ БЛОКА 1 ===

# === БЛОК 2: СНАЙПЕРСКИЙ ПЕРЕХВАТ ЛИНЕЙНЫХ СЛОЕВ И МАРШЕВЫЙ ШАГ ===
# [Этот блок врезает авто-каст типов на уровне линейных операций перемножения весов]
# [и страхует извлечение тензора из кортежа или объекта diffusers]

    base_transformer = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model

    # Массив для фиксации оригинальных методов forward
    orig_linears = []

    # Перехватываем только те линейные слои, которые реально сжаты в FP8
    for name, module in base_transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight") and module.weight.dtype == base_dtype:
            orig_fwd = module.forward
            orig_linears.append((module, orig_fwd))
            
            def make_hybrid_linear_fwd(old_fwd, m_dtype=base_dtype):
                def hybrid_linear_fwd(input_tensor, *args, **kwargs):
                    if input_tensor is not None and input_tensor.dtype != m_dtype:
                        input_tensor = input_tensor.to(dtype=m_dtype)
                    # Выполняем оригинальное перемножение матриц в FP8
                    out = old_fwd(input_tensor, *args, **kwargs)
                    # Мягко возвращаем результат в bfloat16 для остального графа
                    return out.to(dtype=torch.bfloat16)
                return hybrid_linear_fwd
                
            module.forward = make_hybrid_linear_fwd(orig_fwd)

    try:
        # Маршевый проход через изолированный раннер в нативном bf16 контуре
        model_output = lora_model(
            hidden_states=packed_noisy_latents.to(device=device, dtype=meta_dtype),
            timestep=timesteps_attr.to(device=device, dtype=meta_dtype),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=meta_dtype),
            pooled_projections=pooled_projections.to(device=device, dtype=meta_dtype),
            txt_ids=txt_ids.to(device=device, dtype=meta_dtype),
            img_ids=img_ids.to(device=device, dtype=meta_dtype),
            return_dict=False
        )
        
        # Параноидальный шлюз распаковки выхлопа
        if hasattr(model_output, "sample"):
            pred_tensor = model_output.sample
        elif isinstance(model_output, (tuple, list)):
            pred_tensor = model_output[0] if len(model_output) > 0 else model_output
        else:
            pred_tensor = model_output

        # Стабилизация размерности для лосс-вычислителя движка
        if isinstance(pred_tensor, torch.Tensor) and pred_tensor.dim() == 4:
            pred_tensor = pred_tensor.squeeze(1)
            
    finally:
        # Полное восстановление структуры базовых слоев для предотвращения утечек
        for module, orig_fwd in orig_linears:
            module.forward = orig_fwd

    return pred_tensor.to(dtype=meta_dtype)
# === КОНЕЦ БЛОКА 2 ===
