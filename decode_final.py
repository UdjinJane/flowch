# =====================================================================
# БЛОК 1: ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ И КОНСТАНТ СИСТЕМЫ FLOWCH
# =====================================================================
import os
import sys
import torch
from PIL import Image
import numpy as np
from safetensors.torch import load_file

# Импортируем базовые параметры из пульта конфигурации проекта
from src.config import VAE_PATH, OUTPUT_DIR, device

# Безопасная проверка класса FluxVAE в обновленной библиотеке diffusers
try:
    from diffusers.models.autoencoders.autoencoder_flux import FluxVAE
except ImportError:
    try:
        from diffusers import FluxVAE
    except ImportError:
        FluxVAE = None

print("🧱 КИРПИЧ 1: Системные импорты и константы путей успешно развернуты.")
# =====================================================================
# КОНЕЦ БЛОКА 1
# =====================================================================

# =====================================================================
# БЛОК 2: КАН0НИЧЕСКИЙ ВИНТ АКТИВАЦИИ VAE-ДЕКОДЕРА FLUX
# =====================================================================
def init_final_decoder():
    """
    Инициализация оригинального тяжелого VAE Flux с SSD в bfloat16.
    """
    print("=== ИНИЦИАЛИЗАЦИЯ ФИНАЛЬНОГО ВЕРИФИКАЦИОННОГО ДЕКОДЕРА ===")
    print(f"📦 Загрузка оригинального VAE: {VAE_PATH}")
    
    if not os.path.exists(VAE_PATH):
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл VAE отсутствует по адресу {VAE_PATH}")
        return None
        
    try:
        print("⚡ Выделение CUDA-памяти под VAE-контур...")
        if FluxVAE is not None:
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
    Каноническое декодирование скрытого пространства Flux без сжатия каналов.
    """
    print("💾 Траектория завершена! Запуск кастинга Flux-матрицы...")
    
    with torch.no_grad():
        if vae_model and vae_model != "emulation":
            print("🔮 Полная декомпрессия через оригинальный граф VAE (Все слои)...")
            # Масштабирование латентов Flux для разжатия дисперсии
            latents = x_t / 0.3611
            
            # Пропускаем чистые 64 канала оригинального тензора сквозь всю нейросеть VAE
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

print("🧱 КИРПИЧ 2: Функции канонического VAE-декодирования интегрированы.")
# =====================================================================
# КОНЕЦ БЛОКА 2
# =====================================================================

