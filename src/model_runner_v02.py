import torch

class FluxLoRAMarshStep(torch.nn.Module):
    def __init__(self, base_transformer):
        super().__init__()
        self.base = base_transformer
        self.b_dtype = torch.float8_e4m3fn
        self.m_dtype = torch.bfloat16
        
    def patch_blocks(self):
        saved = []
        # Динамический захват двойных блоков
        if hasattr(self.base, "transformer_blocks"):
            for b in self.base.transformer_blocks:
                old_fwd = b.forward
                saved.append((b, old_fwd))
                b.forward = lambda h, e=None, *args, **kwargs: old_fwd(
                    h.to(self.b_dtype) if h is not None else h,
                    e.to(self.b_dtype) if e is not None else e,
                    *args, **kwargs
                )
        # Динамический захват одиночных блоков
        if hasattr(self.base, "single_transformer_blocks"):
            for b in self.base.single_transformer_blocks:
                old_fwd = b.forward
                saved.append((b, old_fwd))
                b.forward = lambda h, *args, **kwargs: old_fwd(
                    h.to(self.b_dtype) if h is not None else h,
                    *args, **kwargs
                )
        return saved

    def forward(self, lora_model, noisy_latents, t_attr, embeds, p_proj, t_ids, i_ids):
        device = noisy_latents.device
        saved_hooks = self.patch_blocks()
        try:
            t_vector = t_attr.flatten() if t_attr is not None else t_attr
            # Чистый позиционный маршевый проход через PEFT-обертку
            out = lora_model(
                noisy_latents.to(device=device, dtype=self.m_dtype),
                t_vector.to(device=device, dtype=self.m_dtype),
                embeds.to(device=device, dtype=self.m_dtype),
                p_proj.to(device=device, dtype=self.m_dtype),
                t_ids.to(device=device, dtype=self.m_dtype),
                i_ids.to(device=device, dtype=self.m_dtype),
                return_dict=False
            )
            pred = out[0] if isinstance(out, tuple) else out
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            return pred.to(dtype=self.m_dtype)
        finally:
            # Железный откат инженерных систем
            for b, old_fwd in saved_hooks:
                b.forward = old_fwd

def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    base_tf = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
    runner = FluxLoRAMarshStep(base_tf)
    return runner(lora_model, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids)
