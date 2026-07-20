# === БЛОК 1: СИСТЕМНЫЕ ИМПОРТЫ И ИНИЦИАЛИЗАЦИЯ РАННЕРА ===
# [Этот блок настраивает окружение и сигнатуру маршевого шага Flux-LoRA]
import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    base_dtype = torch.float8_e4m3fn  # Целевой тип квантованного ядра Flux
    meta_dtype = torch.bfloat16      # Сигнальный тип для эмбеддеров и LoRA
# === КОНЕЦ БЛОКА 1 ===

# === БЛОК 2: ГИБРИДНЫЙ ПЕРЕХВАТ ЛИНЕЙНЫХ СЛОЕВ (FP8 -> BF16) ===
# Реализация гибридного forward: Input(BF16) -> Linear(FP8) -> Output(BF16)
# Перехватывает `torch.nn.Linear` и переводит входы в BF16 перед вычислением.

    base_transformer = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
    orig_linears = []

    # Перехват только `torch.nn.Linear` с FP8 весами
    for name, module in base_transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight") and module.weight.dtype == base_dtype:
            orig_fwd = module.forward
            orig_linears.append((module, orig_fwd))
            
            # Гибридный forward
            def make_hybrid_linear_fwd(old_fwd, m_dtype=base_dtype):
                def hybrid_linear_fwd(input_tensor, *args, **kwargs):
                    if input_tensor is not None and input_tensor.dtype != m_dtype:
                        input_tensor = input_tensor.to(dtype=m_dtype)
                    out = old_fwd(input_tensor, *args, **kwargs)
                    return out.to(dtype=torch.bfloat16)
                return hybrid_linear_fwd
                
            module.forward = make_hybrid_linear_fwd(orig_fwd)

    try:
        # Маршевый проход в нативном bf16 контуре
        model_output = lora_model(
            hidden_states=packed_noisy_latents.to(device=device, dtype=meta_dtype),
            # ... (остальные параметры прохода)
        )
        
        # Обработка выхлопа: возврат Tensor или первый элемент кортежа
        pred_tensor = model_output.sample if hasattr(model_output, "sample") else \
                      (model_output[0] if isinstance(model_output, (tuple, list)) else model_output)

        if isinstance(pred_tensor, torch.Tensor) and pred_tensor.dim() == 4:
            pred_tensor = pred_tensor.squeeze(1)
            
    finally:
        # Восстановление слоев
        for module, orig_fwd in orig_linears:
            module.forward = orig_fwd

    return pred_tensor.to(dtype=meta_dtype)
# === КОНЕЦ БЛОКА 2 ===

