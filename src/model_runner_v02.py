import torch

class FluxLoRAMarshStep(torch.nn.Module):
    def __init__(self, base_transformer):
        super().__init__()
        self.base = base_transformer
        self.b_dtype = torch.float8_e4m3fn
        # self.m_dtype = torch.bfloat16

    def patch_blocks(self):
        saved = []
        return []
        # Динамический захват двойных блоков
        if hasattr(self.base, "transformer_blocks"):
            for b in self.base.transformer_blocks:
                old_fwd = b.forward
                saved.append((b, old_fwd))
                # Насильно держим входящие состояния в bfloat16 для безопасного torch.cat
                b.forward = lambda hidden_states, encoder_hidden_states=None, *args, **kwargs: old_fwd(
                    hidden_states.to(self.m_dtype) if hidden_states is not None else hidden_states,
                    encoder_hidden_states.to(self.m_dtype) if encoder_hidden_states is not None else encoder_hidden_states,
                    *args, **kwargs
                )
        # Динамический захват одиночных блоков
        if hasattr(self.base, "single_transformer_blocks"):
            for b in self.base.single_transformer_blocks:
                old_fwd = b.forward
                saved.append((b, old_fwd))
                # Удерживаем в bfloat16
                b.forward = lambda hidden_states, *args, **kwargs: old_fwd(
                    hidden_states.to(self.m_dtype) if hidden_states is not None else hidden_states,
                    *args, **kwargs
                )
        return saved


        def forward(self, lora_model, noisy_latents, t_attr, embeds, p_proj, t_ids, i_ids):
            device = noisy_latents.device
            # Отключаем ручной перехват блоков, так как переходим на системный autocast
            try:
                if t_attr is not None:
                    t_vector = t_attr.reshape(-1)[:noisy_latents.shape[0]]
                else:
                    t_vector = t_attr

                # --- ОДНОКРАТНАЯ ТЕЛЕМЕТРИЯ ---
                if not hasattr(self, "_telemetry_fired"):
                    print("\n" + "="*50)
                    print("[ТЕЛЕМЕТРИЯ МОСТИКА] Вход по именованному контуру:")
                    print(f" -> hidden_states: {list(noisy_latents.shape)}")
                    print("="*50 + "\n")
                    self._telemetry_fired = True

                # Запуск системного выравнивателя типов для bfloat16/float8 матриц
                with torch.amp.autocast(device_type="cuda", dtype=self.m_dtype):
                    out = lora_model(
                        hidden_states=noisy_latents.to(device=device, dtype=self.m_dtype),
                        timestep=t_vector.to(device=device, dtype=self.m_dtype) if t_vector is not None else None,
                        encoder_hidden_states=embeds.to(device=device, dtype=self.m_dtype),
                        pooled_projections=p_proj.to(device=device, dtype=self.m_dtype),
                        txt_ids=t_ids.to(device=device, dtype=self.m_dtype),
                        img_ids=i_ids.to(device=device, dtype=self.m_dtype),
                        return_dict=False
                    )
                    
                pred = out[0] if isinstance(out, tuple) else out
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
                return pred.to(dtype=self.m_dtype)
                
            finally:
                pass


def run_lora_model_step(lora_model, batch, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids):
    base_tf = lora_model.get_base_model() if hasattr(lora_model, "get_base_model") else lora_model
    runner = FluxLoRAMarshStep(base_tf)
    return runner(lora_model, packed_noisy_latents, timesteps_attr, prompt_embeds, pooled_projections, txt_ids, img_ids)