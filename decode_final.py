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
