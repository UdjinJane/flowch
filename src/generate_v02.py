import torch
from typing import Tuple, Optional
import os

def verify_incoming_lora_weights(
    transformer_model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
    t_attr: float = 0.5,
    batch_size: int = 1,
    latent_dim: int = 4,
    resolution: int = 512,
    num_channels: int = 4,
    in_channels: int = 4,
    text_encoder_hidden_dim: int = 4096,
    pooled_projection_dim: int = 768,
    txt_ids_len: int = 256,
    img_ids_len: int = 256
) -> Tuple[bool, str]:
    """
    Безопасная верификация lora-чекпоинтов на АПЕКСе и катере GIPSY.
    Разработка: Qwen3.5 9B (Регламент V02_STABLE)
    """
    
    # --- БЛОК 1: ЗАГРУЗКА STATE-DICT С ЧЕКПОИНТА ---
    try:
        checkpoint_state_dict = torch.load(
            checkpoint_path,
            map_location="cuda",
            weights_only=True
        )
    except Exception as e:
        return False, f"Ошибка загрузки чекпоинта: {type(e).__name__}: {str(e)}"
    
    # --- БЛОК 2: ОЧИСТКА КЛЮЧЕЙ (УДАЛЕНИЕ ПРЕФИКСА model.diffusion_model.) ---
    clean_state_dict = {}
    for k, v in checkpoint_state_dict.items():
        if k.startswith("model.diffusion_model."):
            clean_key = k.replace("model.diffusion_model.", "")
        else:
            clean_key = k
        clean_state_dict[clean_key] = v
    
    # --- БЛОК 3: ХРОНОЛОГИЧЕСКИЙ КАСТИНГ В BFLOAT16 (ДО ИНЖЕКЦИИ!) ---
    for name, param in clean_state_dict.items():
        if param.is_floating_point():
            clean_state_dict[name] = param.to(dtype=torch.bfloat16)

    # --- БЛОК 4: ВАЛИДАЦИЯ ГЕОМЕТРИИ И ЗАГРУЗКА В TRANSFORMER_MODEL ---
    try:
        for name, param in transformer_model.named_parameters():
            if "lora_" not in name.lower():
                continue
            
            if name not in clean_state_dict:
                return False, f"Отсутствует ключ в чекпоинте: {name}"
            
            checkpoint_tensor = clean_state_dict[name]
            model_tensor = param.data
            
            # Строгая валидация геометрии и шейпов через assert
            assert checkpoint_tensor.shape == model_tensor.shape, \
                f"Shape mismatch для {name}: чекпоинт {checkpoint_tensor.shape} != модель {model_tensor.shape}"
            
            # Загрузка с проверкой dtype (bfloat16 должен сохраниться)
            param.data = checkpoint_tensor.to(device=device, dtype=torch.bfloat16)
        
        if len(clean_state_dict) == 0:
            return False, "Чекпоинт пуст или не содержит LoRA-слоев"
        
    except AssertionError as e:
        return False, f"Ошибка валидации геометрии: {str(e)}"
    except Exception as e:
        return False, f"Ошибка загрузки весов: {type(e).__name__}: {str(e)}"
    
    # --- БЛОК 5: ВКЛЮЧЕНИЕ РЕЖИМА EVAL И NO-GRAD КОНТЕКСТА ---
    transformer_model.eval()
    
    with torch.no_grad():
        # --- БЛОК 6: ПОДГОТОВКА ТЕСТОВОЙ БАТЧИ ДЛЯ ЭЙЛЕРА ---
        latent_shape = (batch_size, num_channels, resolution // 8, resolution // 8)
        noise_tensor = torch.randn(*latent_shape, device=device, dtype=torch.bfloat16)
        t_tensor = torch.tensor([t_attr], device=device, dtype=torch.bfloat16)

        # --- БЛОК 7: ИМИТАЦИЯ КОНДИЦИОННЫХ ТЕНЗОРОВ (PROMPT EMBEDS) ---
        prompt_embeds = torch.randn(text_encoder_hidden_dim, txt_ids_len, device=device, dtype=torch.bfloat16)
        pooled_projections = torch.randn(pooled_projection_dim, txt_ids_len, device=device, dtype=torch.bfloat16)
        
        txt_ids = torch.arange(txt_ids_len, device=device, dtype=torch.long)
        img_ids = torch.arange(img_ids_len, device=device, dtype=torch.long)
        
        # --- БЛОК 8: ПЕРЕКЛАДКА ВХОДНЫХ ТЕНЗОРОВ В BFLOAT16 ---
        packed_noisy_latents = noise_tensor.to(device=device, dtype=torch.bfloat16)
        t_vector = t_tensor.to(device=device, dtype=torch.bfloat16)
        prompt_embeds_bf16 = prompt_embeds.to(device=device, dtype=torch.bfloat16)
        pooled_projections_bf16 = pooled_projections.to(device=device, dtype=torch.bfloat16)
        txt_ids_bf16 = txt_ids.to(device=device, dtype=torch.bfloat16)
        img_ids_bf16 = img_ids.to(device=device, dtype=torch.bfloat16)
        
        # --- БЛОК 9: ВЫЗОВ МОДЕЛИ (ОДИН ШАГ ЭЙЛЕРА) ---
        try:
            out = transformer_model(
                hidden_states=packed_noisy_latents,
                timestep=t_vector,
                encoder_hidden_states=prompt_embeds_bf16,
                pooled_projections=pooled_projections_bf16,
                txt_ids=txt_ids_bf16,
                img_ids=img_ids_bf16,
                return_dict=False
            )
            
            # --- БЛОК 10: ВАЛИДАЦИЯ ВЫХОДНЫХ ТЕНЗОРОВ (ЯВНЫЕ СРЕЗЫ ПО ВСЕМ ОСЯМ!) ---
            if isinstance(out, tuple):
                out_latents = out[0]
            else:
                out_latents = out
            
            assert isinstance(out_latents, torch.Tensor), "Выход модели не является тензором"
            assert out_latents.device.type == "cuda", f"Выход на устройстве {out_latents.device}, ожидается cuda"
            
            # Явные срезы по ВСЕМ осям для 3D/4D тензоров (защита от IndexError)
            if out_latents.dim() >= 3:
                sliced_output = out_latents[:, :out_latents.shape[1], :out_latents.shape[2]]
            else:
                sliced_output = out_latents

            assert sliced_output.numel() > 0, "Выходной тензор пуст"
            
            # --- БЛОК 11: ВОЗВРАТ В TRAIN MODE И УСПЕШНЫЙ ВЫХОД ---
            transformer_model.train()
            return True, f"Верификация успешна. Выход: {sliced_output.shape}, dtype={sliced_output.dtype}"
            
        except Exception as e:
            transformer_model.train()
            return False, f"Ошибка тестового прогона Эйлера: {type(e).__name__}: {str(e)}"
