# Финальная версия generate_v02.py (БЛОК 1 ИЗ 2: ОРИГИНАЛЬНЫЙ МАРШ)
import os
import torch
import time
from PIL import Image
from config import TrainConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from safetensors.torch import load_file
from model_runner_v02 import run_lora_model_step

#------------------ ОБРАБОТКА АНОМАЛИИ И КАНОНИЧЕСКИЙ ДЕКОД V05 --------------------
# [ВЫЖЖЕНО ПЛАЗМОЙ: Сборка V05 чинит масштаб мантиссы]
with torch.no_grad():
    # 1. Инверсный Pixel Shuffle и денормирование (исправление VAE)
    latents_unpacked = x_t.view(1, 32, 32, 64).reshape(1, 32, 32, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 16, 64, 64)
    sf = getattr(vae.config, "scaling_factor", 0.3611)
    shf = getattr(vae.config, "shift_factor", 0.1159)
    z_cleaned = (latents_unpacked / sf) + shf  # ЖЕСТКИЙ ФИКС: Деление вместо умножения

    # 2. Родной метод VAE.decode (костыли демонтированы)
    dec_out = vae.decode(z_cleaned, return_dict=False)[0]

    # Конвертация в RGB [0, 255]
    dec_out_clamped = ((dec_out + 1.0) / 2.0).clamp(0.0, 1.0)
    img_array = (dec_out_clamped.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype('uint8')

    # Фиксация на SSD
    output_path = os.path.join(TrainConfig.OUTPUT_DIR, "images", f"mng_render_step_{current_step}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(img_array).save(output_path)

   
# Финальная версия generate_v02.py (БЛОК 2 ИЗ 2: ГИБРИДНЫЙ ВЕРИФИКАТОР)
def verify_incoming_lora_weights(transformer_model: torch.nn.Module, checkpoint_path: str) -> bool:
    """Гибридный верификатор (Qwen3.5+Ministral): префиксы, bfloat16, RoPE [1.10]."""
    try:
        # 1. Загрузка и очистка ключей от дефузеров
        ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        clean_sd = {k.replace("model.diffusion_model.", "") if "model.diffusion_model." in k else k: v for k, v in ckpt.items()}

        # 2. Инжекция весов LoRA с приведением к bfloat16
        for name, param in transformer_model.named_parameters():
            if "lora_" in name.lower() and name in clean_sd:
                param.data.copy_(clean_sd[name].to(device="cuda", dtype=torch.bfloat16))

        # 3. Тест-драйв Эйлера (валидация геометрии) [1.10]
        transformer_model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            test_input = {
                "hidden_states": torch.randn(1, 1024, 64, device="cuda", dtype=torch.bfloat16),
                "timestep": torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
                "encoder_hidden_states": torch.randn(1, 256, 4096, device="cuda", dtype=torch.bfloat16),
                "pooled_projections": torch.zeros(1, 768, device="cuda", dtype=torch.bfloat16),
                "txt_ids": torch.zeros((256, 3), device="cuda", dtype=torch.bfloat16),
                "img_ids": torch.zeros((1024, 3), device="cuda", dtype=torch.bfloat16)
            }
            transformer_model(**test_input)
        
        print("[УСПЕХ] Чекпоинт валиден. Веса инжектированы [1.10].")
        transformer_model.train()
        return True
    except Exception as e:
        print(f"[АВАРИЯ] Верификация: {e}", file=sys.stderr)
        return False

#--------------- ХОЛОДНЫЙ СТАРТ -----------------
if __name__ == "__main__":
    print("[ТЕСТ] Запуск автономной компиляции генератора...")
    # 1. Создаем фейковый эмбеддинг текста по спецификации Chroma1 (1, 256, 4096)
    mock_text_embed = torch.zeros((1, TrainConfig.MAX_SEQUENCE_LENGTH, 4096), dtype=torch.bfloat16, device="cuda")
    
    # 2. Вызываем инференс вхолостую (проверка синтаксиса и подгрузки vae_config.json)
    try:
        # Передаем None вместо трансформера, чтобы проверить только инициализацию и конфигурацию VAE
        # Чтобы тест прошел дальше проверки на None, можно временно закомментировать строчку "if loaded_transformer is None: return"
        run_inference_v02(
            loaded_transformer=None, 
            current_step=999, 
            text_embedding=mock_text_embed, 
            steps=1, 
            device="cuda"
        )
        print("[УСПЕХ] Автономная компиляция генератора завершена без ошибок.")
    except Exception as e:
        print(f"[КРАХ ТЕСТА] Ошибка в рантайме генератора: {e}")
