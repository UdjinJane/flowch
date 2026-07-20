import torch

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    device = torch.device("cuda")
    meta_dtype = torch.bfloat16

    if timesteps_attr is not None:
        t_vector = timesteps_attr.reshape(-1)[:packed_noisy_latents.shape[0]]
    else:
        t_vector = timesteps_attr

    if not hasattr(run_lora_model_step, "_telemetry_fired"):
        print("\n" + "="*50)
        print("[ТЕЛЕМЕТРИЯ МОСТИКА] Вход по прямому функциональному контуру:")
        print(f" -> hidden_states: {list(packed_noisy_latents.shape)}")
        print("="*50 + "\n")
        run_lora_model_step._telemetry_fired = True

    with torch.amp.autocast(device_type="cuda", dtype=meta_dtype):
        out = lora_model(
            hidden_states=packed_noisy_latents.to(device=device, dtype=meta_dtype),
            timestep=t_vector.to(device=device, dtype=meta_dtype) if t_vector is not None else None,
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=meta_dtype),
            pooled_projections=pooled_projections.to(device=device, dtype=meta_dtype),
            txt_ids=txt_ids.to(device=device, dtype=meta_dtype),
            img_ids=img_ids.to(device=device, dtype=meta_dtype),
            return_dict=False
        )

    pred_tensor = out if isinstance(out, tuple) else out
    if pred_tensor.dim() == 4:
        pred_tensor = pred_tensor.squeeze(1)

    return pred_tensor.to(dtype=meta_dtype)
