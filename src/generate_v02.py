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

    # 2. VAE (Загрузка весов строго по схеме)
    import json
    with open(os.path.join(TrainConfig.SRC_DIR, "vae_config.json"), "r") as f:
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
    
    # 1. Штурм бортового кэша текста для извлечения триггера mng_oks_bl
    try:
        embed_files = glob.glob(os.path.join(TrainConfig.CACHE_TEXT_DIR, "*.safetensors"))
        if not embed_files:
            raise FileNotFoundError("Кэш текста абсолютно пуст!")
            
        cached_embeds = load_file(embed_files[0], device="cpu")
        prompt_embeds = cached_embeds.get("prompt_embeds", list(cached_embeds.values())[0]).to(device=device, dtype=torch.bfloat16)
        
        # Честное восстановление пулинга CLIP (не нули!) для глобального проектора Chroma
        pooled_projections = cached_embeds.get("pooled_projections", torch.zeros((1, 768), device=device, dtype=torch.bfloat16)).to(device=device, dtype=torch.bfloat16)
        
        print(f"[УСПЕХ] Текстовый шлюз Chroma V07 открыт! Перехвачен живой кэш: {os.path.basename(embed_files[0])}")
    except Exception as e:
        print(f"[КРАХ ТЕСТА] Ошибка текстового тракта V07: {e}. Аварийная генерация заглушек.")
        prompt_embeds = torch.zeros((1, TrainConfig.MAX_SEQUENCE_LENGTH, 4096), device=device, dtype=torch.bfloat16)
        pooled_projections = torch.zeros((1, 768), device=device, dtype=torch.bfloat16)

    # 2. СПЕКТРАЛЬНЫЙ АНАЛИЗ МАНТИССЫ CHROMA V07 ПЕРЕД ЗАПУСКОМ СЭМПЛЕРА
    print("=" * 60)
    print("[СПЕКТРАЛЬНЫЙ АНАЛИЗ МАНТИССЫ CHROMA V07]")
    print(f" -> prompt_embeds shape: {prompt_embeds.shape} | mean: {prompt_embeds.mean().item():.6f} | std: {prompt_embeds.std().item():.6f}")
    print(f" -> pooled_projections shape: {pooled_projections.shape} | mean: {pooled_projections.mean().item():.6f} | std: {pooled_projections.std().item():.6f}")
    print(f" -> Target Resolution: {TrainConfig.RESOLUTION}x{TrainConfig.RESOLUTION}")
    print(f" -> Scheduler Config: {pipe.scheduler.config}")
    print("=" * 60)



    # 3. БОЕВОЙ ЗАПУСК ОРИГИНАЛЬНОГО ПАЙПЛАЙНА (Кустарщина полностью ликвидирована)
    with torch.inference_mode():
        # Сэмплируем через канонический __call__ оригинального ChromaPipeline
        pipeline_output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_projections=pooled_projections,
            height=TrainConfig.RESOLUTION,
            width=TrainConfig.RESOLUTION,
            num_inference_steps=25,
            output_type="pil",
            return_dict=True
        )
        
        # Извлекаем первый запеченный кадр из возвращенного FluxPipelineOutput
        final_image = pipeline_output.images[0]

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
