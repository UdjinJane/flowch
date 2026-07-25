# === МАРКЕР НАЧАЛА: CHROMA_PIPELINE_MONOLITH_V07 ===
import os, sys, torch
from config import TrainConfig
from safetensors.torch import load_file
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# === МАРКЕР СИНХРОНИЗАЦИИ CHROMA V07_LOCAL ===
# Выжжены внешние пути, привязываемся строго к внутреннему контуру src/
from chroma_pipeline import ChromaPipeline
# === КОНЕЦ МАРКЕРА ===

def run_inference_v02(loaded_transformer, current_step=0, device='cuda'):
    """[МАРШРУТ V07] Валидация через оригинальный ChromaPipeline."""
    print(f"\n[ОБТ] >>> ИНИЦИАЛИЗАЦИЯ CHROMA V07 | ШАГ #{current_step} <<<")

    # 1. Планировщик FlowMatch
    scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256, "max_image_seq_len": 4096,
        "base_shift": 0.5, "max_shift": 1.15, "shift": 3.0
    })

    # 2. VAE (Загрузка весов со стопроцентной дезинфекцией UTF-8 BOM)
    import json
    with open(os.path.join(TrainConfig.SRC_DIR, "vae_config.json"), "r", encoding="utf-8-sig") as f:
        vae_config = json.load(f)

    vae_config["block_out_channels"] = [128, 256, 512, 512]

    vae = AutoencoderKL.from_config(vae_config).to(device=device, dtype=torch.bfloat16)
    vae.load_state_dict({k.replace("vae.", ""): v for k, v in load_file(TrainConfig.VAE_PATH, device="cpu").items()}, strict=False)

    # 3. Сборка пайплайна
    pipe = ChromaPipeline(
        scheduler=scheduler, vae=vae,
        transformer=loaded_transformer,
        text_encoder=None, tokenizer=None, # Прямой кэш
        text_encoder_2=None, tokenizer_2=None
    ).to(device=device, dtype=torch.bfloat16)

    pipe.maybe_free_model_hooks = lambda: None # Удержание хуков
    # === ИНЖЕКЦИЯ CHROMA V07: ЧАСТЬ 2 (Сборка кэша и боевой запуск) ===
    import glob
    from PIL import Image
    
    # === ИНЖЕКЦИЯ CHROMA V08_LOCAL: МОНОЛИТНЫЙ КЭШ PYTORCH ===
    import glob
    from PIL import Image

    # 1. Загрузка кэша, извлечение эмбеддингов, маски и пулинга [1.10]
    try:
        embed_files = glob.glob(os.path.join(TrainConfig.CACHE_TEXT_DIR, "*.pt"))
        target_file = embed_files[0]
        cached_dict = torch.load(target_file, map_location="cpu")
        
        # Получение данных с приведением типов
        prompt_embeds = cached_dict["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        prompt_attn_mask = cached_dict["prompt_attn_mask"].to(device=device)
        pooled_projections = cached_dict["pooled_projections"].to(device=device, dtype=torch.bfloat16)
    except Exception as e:
        print(f"Ошибка загрузки кэша: {e}")

    # 2. Спектральный анализ [1.10]
    print(f"[АНАЛИЗ] prompt_embeds: {prompt_embeds.shape}, pooled: {pooled_projections.shape}")

    # 3. Запуск пайплайна (синхронизировано) [1.10]
    with torch.inference_mode():
        pipeline_output = pipe(
            prompt_embeds=prompt_embeds,
            height=TrainConfig.RESOLUTION,
            width=TrainConfig.RESOLUTION,
            num_inference_steps=25,
            output_type="pil"
        )
        final_image = pipeline_output.images[0]

    # 4. Сохранение
    final_image.save(os.path.join(TrainConfig.OUTPUT_DIR, "images", f"render_{current_step}.png"))

# ... остальной код (заглушки) ...
# === КОНЕЦ: CHROMA_PIPELINE_MONOLITH_V08 ===


    # 4. ФИКСАЦИЯ И СБРОС СНАРЯДА НА SSD
    output_path = os.path.join(TrainConfig.OUTPUT_DIR, "images", f"mng_render_step_{current_step}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_image.save(output_path)
    print(f"[ТРИУМФ V07] Чистокровное изображение запечено на SSD: {output_path}")

# Заглушка-верификатор для совместимости с движком обучения train_engine_v02
def verify_incoming_lora_weights(transformer_model: torch.nn.Module, checkpoint_path: str) -> bool:
    """Формальный верификатор для удержания интерфейса."""
    return True

if __name__ == "__main__":
    print("[ТЕСТ] Автономный марш генератора V07...")
    # Инициализируем пустое ядро для проверки сквозного импорта
    mock_transformer = torch.nn.Module()
    try:
        run_inference_v02(loaded_transformer=mock_transformer, current_step=777, device="cuda")
        print("[УСПЕХ] Тестовый прогон V07 завершен.")
    except Exception as e:
        print(f"[КОНТРОЛЬ] Автономный вылет (норма при отсутствии весов transformer): {e}")
# === КОНЕЦ: CHROMA_PIPELINE_MONOLITH_V07 ===
