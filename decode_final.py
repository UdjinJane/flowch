import os
import sys
import torch
from PIL import Image
import numpy as np
from safetensors.torch import load_file

# Импортируем базовые константы из пульта управления проектом
from src.config import VAE_PATH, OUTPUT_DIR, device

# Динамический импорт оригинального класса FluxVAE под новые библиотеки
try:
    from diffusers.models.autoencoders.autoencoder_flux import FluxVAE
except ImportError:
    try:
        from diffusers import FluxVAE
    except ImportError:
        FluxVAE = None

def init_final_decoder():
    """
    Блок 1: Загрузка оригинального тяжелого VAE Flux с SSD через правильный класс.
    """
    print("=== ИНИЦИАЛИЗАЦИЯ ФИНАЛЬНОГО ВЕРИФИКАЦИОННОГО ДЕКОДЕРА ===")
    print(f"📦 Загрузка оригинального VAE: {VAE_PATH}")
    
    if not os.path.exists(VAE_PATH):
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл VAE отсутствует по адресу {VAE_PATH}")
        return None
        
    try:
        print("⚡ Выделение CUDA-памяти под VAE-контур...")
        if FluxVAE is not None:
            # Загружаем веса Flux VAE в bfloat16 для RTX 3090
            vae_model = FluxVAE.from_single_file(VAE_PATH, torch_dtype=torch.bfloat16).to(device)
            vae_model.eval()
            print("✅ Настоящий VAE Flux успешно развернут в памяти девайса!")
            return vae_model
        else:
            print("⚠ Класс FluxVAE недоступен в текущей версии diffusers. Переход в эмуляцию.")
            return "emulation"
    except Exception as e:
        print(f"⚠ Сбой инициализации класса VAE: {e}")
        return "emulation"
def decode_latents_to_rgb(x_t, vae_model, output_path):
    """
    Блок 2: Полноценное каноническое декодирование скрытого пространства Flux.
    """
    print("💾 Траектория завершена! Запуск кастинга Flux-матрицы...")
    
    with torch.no_grad():
        if vae_model and vae_model != "emulation":
            print("🔮 Полная декомпрессия через оригинальный граф VAE (Все слои)...")
            # Каноническое масштабирование латентов Flux для разжатия дисперсии
            latents = x_t / 0.3611
            
            # Скармливаем оригинальные 64 канала — обновленный FluxVAE сделает всё сам
            output_tensor = vae_model.decode(latents).sample
            image_tensor = torch.clamp((output_tensor + 1.0) / 2.0, 0.0, 1.0)
        else:
            print("⚠ Режим эмуляции: кастинг матрицы через апскейл среднего значения...")
            x_t_vis = x_t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            image_tensor = (x_t_vis - x_t_vis.min()) / (x_t_vis.max() - x_t_vis.min() + 1e-5)
        
        # Перевод тензора PyTorch [1, 3, H, W] -> PIL [H, W, 3]
        img_array = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8')
        final_img = Image.fromarray(img_array)
        
        if final_img.size != (512, 512):
            final_img = final_img.resize((512, 512), Image.Resampling.BILINEAR)
            
        final_img.save(output_path)
        print(f"🎉 ФИНАЛЬНЫЙ ЦВЕТНОЙ РЕНДЕР СФОРМИРОВАН: {output_path}")

def run_final_inference(checkpoint_path, vae_model, epoch=150, text_embedding=None):
    """
    Блок 3: Сборка инференса с жестким бронированным абсолютным путем сохранения.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ ОШИБКА: Чекпоинт {checkpoint_path} не найден.")
        return

    try:
        from src.models import EmptyTransformer
        from src.model_utils import inject_chroma_lora
        
        transformer = EmptyTransformer().to(device)
        transformer = inject_chroma_lora(transformer)
        
        print(f"🚀 Загрузка весов LoRA из чекпоинта: {checkpoint_path}")
        lora_sd = load_file(checkpoint_path)
        transformer.load_state_dict(lora_sd, strict=False)
        transformer.eval()
    except Exception as e:
        print(f"❌ Сбой подготовки модели: {e}")
        return

    # Жесткий абсолютный путь в обход кривых локальных путей
    output_dir = r"Z:\flowch\output\images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mng_final_vae_epoch_{epoch}.png")
    
    print("⚡ Контур инференса готов к финишному рендеру.")
    
    x_t = torch.randn(1, 64, 64, 64, device=device)
    steps = 25
    dt = 1.0 / steps
    
    with torch.no_grad():
        for i in range(steps):
            t_curr = i * dt
            t_tensor = torch.ones(1, device=device) * t_curr
            
            if text_embedding is not None:
                velocity = transformer(x_t, t_tensor, text_embedding.to(device))
            else:
                velocity = transformer(x_t, t_tensor)
                
            x_t = x_t + velocity * dt
            if (i + 1) % 5 == 0 or (i + 1) == steps:
                print(f"  [~] Прогресс ODE: {int(((i + 1) / steps) * 100)}%")

    decode_latents_to_rgb(x_t, vae_model, output_path)

if __name__ == "__main__":
    vae_model = init_final_decoder()
    if vae_model is not None:
        target_check = r"Z:\flowch\checkpoints\chroma1_mangala_lora_epoch_150.safetensors"
        
        val_emb = None
        try:
            from src.models import ChromaDataset
            dataset = ChromaDataset()
            val_emb = dataset['t5_hidden'].unsqueeze(0).to(device)
            print("🎯 Валидационный текстовый компас T5-XXL успешно подключен!")
        except Exception:
            print("⚠ Текстовый компас не найден. Запуск в режиме базового контекста.")

        run_final_inference(target_check, vae_model, epoch=150, text_embedding=val_emb)
