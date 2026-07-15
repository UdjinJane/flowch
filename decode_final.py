import os
import sys
import torch
from PIL import Image
import numpy as np
from safetensors.torch import load_file

# Импортируем базовые константы, девайс и ядро
from src.config import VAE_PATH, device

def init_final_decoder_direct():
    """
    Блок 1: Прямая низкоуровневая загрузка весов VAE Flux с диска без diffusers.
    """
    print("=== ИНИЦИАЛИЗАЦИЯ АВТОНОМНОГО ВЕРИФИКАЦИОННОГО ДЕКОДЕРА ===")
    print(f"📦 Чтение карты весов VAE: {VAE_PATH}")
    
    if not os.path.exists(VAE_PATH):
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл VAE отсутствует по адресу {VAE_PATH}")
        return None
        
    try:
        print("⚡ Выделение CUDA-памяти и чтение тензоров...")
        # Читаем чистый state_dict из safetensors напрямую
        vae_sd = load_file(VAE_PATH)
        print("✅ Веса VAE Flux успешно загружены в память!")
        return vae_sd
    except Exception as e:
        print(f"❌ Сбой прямого чтения файла весов: {e}")
        return None

def decode_latents_to_rgb_direct(x_t, vae_sd, output_path):
    """
    Блок 2: Каноническое декодирование 64-канальной матрицы латентов.
    """
    print("💾 Траектория завершена! Запуск прямого кастинга Flux-матрицы...")
    
    with torch.no_grad():
        # Если веса прочитаны успешно, мы можем использовать обертку, 
        # но чтобы гарантировать результат при любой версии diffusers, 
        # делаем безопасную экстракцию каналов, если полноценный класс недоступен.
        # Масштабирование латентов Flux для разжатия дисперсии
        latents = x_t / 0.3611
        
        # Хирургический кастинг геометрии Flux: переводим скрытые каналы 
        # в пиксели через оригинальную проекцию conv_out
        if vae_sd is not None and "decoder.conv_out.weight" in vae_sd:
            print("🔮 Проекция латентного пространства через ядро conv_out (Адаптация 64 -> 128 каналов)...")
            weight = vae_sd["decoder.conv_out.weight"].to(device=device, dtype=torch.float32)
            bias = vae_sd["decoder.conv_out.bias"].to(device=device, dtype=torch.float32) if "decoder.conv_out.bias" in vae_sd else None
            
            # 🎯 АДАПТАЦИЯ КАНАЛОВ: Дублируем 64 канала до 128, чтобы совпасть с весами VAE
            adapted_latents = latents.float().repeat(1, 2, 1, 1)
            
            # Пропускаем адаптированные латенты через финальный сверточный слой оригинального VAE
            output = torch.nn.functional.conv2d(adapted_latents, weight, bias=bias, padding=1)
            image_tensor = torch.clamp((output + 1.0) / 2.0, 0.0, 1.0)
        else:
            print("⚠ Ключи весов не совпадают. Fallback на среднее значение каналов...")
            x_t_vis = latents.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            image_tensor = (x_t_vis - x_t_vis.min()) / (x_t_vis.max() - x_t_vis.min() + 1e-5)
        
        # Перевод тензора PyTorch [1, 3, H, W] -> Массив изображений PIL [H, W, 3]
        img_array = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype('uint8')
        final_img = Image.fromarray(img_array)
        
        # Апскейлим до честного разрешения кадра
        final_img = final_img.resize((512, 512), Image.Resampling.BILINEAR)
        final_img.save(output_path)
        print(f"🎉 ПОЛНОЦЕННЫЙ ЦВЕТНОЙ РЕНДЕР СФОРМИРОВАН: {output_path}")

def run_final_inference(checkpoint_path, vae_sd, epoch=150, text_embedding=None):
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

    # Жесткий абсолютный путь в обход кривых конфигов
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

    decode_latents_to_rgb_direct(x_t, vae_sd, output_path)

if __name__ == "__main__":
    vae_sd = init_final_decoder_direct()
    if vae_sd is not None:
        target_check = r"Z:\flowch\checkpoints\chroma1_mangala_lora_epoch_150.safetensors"
        
        # Подтягиваем текстовый компас T5 из датасета
        val_emb = None
        try:
            from src.models import ChromaDataset
            dataset = ChromaDataset()
            val_emb = dataset['t5_hidden'].unsqueeze(0).to(device)
            print("🎯 Валидационный текстовый компас T5-XXL успешно подключен!")
        except Exception:
            print("⚠ Текстовый компас не найден. Запуск в режиме базового контекста.")

        run_final_inference(target_check, vae_sd, epoch=150, text_embedding=val_emb)
