# =====================================================================
# ПУЛЬТ УПРАВЛЕНИЯ РЕАКТОРОМ: КОНФИГУРАЦИЯ ПРОЕКТА "CHROMA-PECH 2.0"
# =====================================================================
import os
import torch

# ⚡ Окружение и системные флаги оптимизации CUDA
os.environ['TORCH_CUDNN_V_GTE_9'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 📂 МАГИСТРАЛИ ДАННЫХ И ДАТАСЕТА
DATASET_DIR = r'Z:\flowch\dataset\mng_oks_bl'
METADATA_PATH = r'Z:\flowch\metadata.jsonl'
OUTPUT_DIR = r'Z:\flowch'

# 📦 КРЕМНИЕВЫЕ ЯДРА И МОДЕЛИ (Chroma1-Base & Текстовый контур)
# Главный базовый чекпоинт Chroma1-Base (fp16 монолит со встроенным CLIP)
MODEL_BASE_PATH = r'Z:\AiModels\models\diffusion_models\Chroma1-Base.safetensors'

# Наш главный текстовый компас T5-XXL в энергоэффективном формате fp8
T5_XXL_PATH = r'Z:\AiModels\models\clip\t5xxl_fp8_e4m3fn.safetensors'

# Оригинальный тяжелый VAE-декодер для финишной распаковки латентов в RGB
VAE_PATH = r'Z:\AiModels\models\vae\flux_vae.safetensors'

# 📐 ПАРАМЕТРЫ ПЛАВКИ И ГЕОМЕТРИИ ЛАТЕНТОВ
# Эталонное разрешение базовой Chroma1-Base для идеальной сходимости LoRA
TRAIN_RESOLUTION = 512

# Оптимальный размер батча для жесткого удержания VRAM в пределах 16.4 ГБ на RTX 3090
batch_size = 1

# Длина тренировочной траектории (количество эпох)
num_epochs = 150

# ⚙ СПЕЦИФИКАЦИЯ АРХИТЕКТУРЫ CHROMA1 (Для новых скриптов)
TIMESTEP_SAMPLING = "x^2"    # Квадратичный сдвиг распределения шагов Rectified Flow
MASK_PADDING_TOKENS = True   # Принудительная активация MMDiT маскирования Т5-паддингов
