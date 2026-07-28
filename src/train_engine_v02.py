# === НАЧАЛО СЛУЖЕБНОГО КУСКА №1: ВЕРХНИЙ ЖЕЛЕЗНЫЙ ШЛЮЗ ===
import os
import sys
import gc

# Динамическая навигация по полкам проекта
_путь_к_файлу = os.path.abspath(__file__)
_папка_src = os.path.dirname(_путь_к_файлу)
_корень_проекта = os.path.dirname(_папка_src)

# Заталкиваем наши отсеки в самый верх системного поиска
for _адрес in [_папка_src, _корень_проекта]:
    if _адрес not in sys.path:
        sys.path.insert(0, _адрес)

# ХАК ОМНИССИИ: Спасаем переименованную математику от коллизий с системой
import chroma_math
sys.modules["src.math"] = chroma_math

# ЖЕСТКОЕ ПРЕДВАРИТЕЛЬНОЕ ВЫЖИГАНИЕ AMD-ГРЯЗИ
os.environ.update({
    "QUANTO_DISABLE_CPP_EXT": "1", 
    "HF_DISABLE_COMPILING": "1", 
    "FORCE_CUDA": "1", 
    "USE_ROCM": "0"
})
for _переменная in ["ROCM_HOME", "HIP_PATH", "OLLAMA_LLM_LIBRARY", "ROCM_PATH"]: 
    os.environ.pop(_переменная, None)
# === КОНЕЦ СЛУЖЕБНОГО КУСКА №1 ===
# === НАЧАЛО СЛУЖЕБНОГО КУСКА №2: МАРШЕВЫЕ ИМПОРТЫ И ВАКУУМ ПАМЯТИ ===
import torch
import diffusers.utils.import_utils as du
from torch.optim import AdamW
from config import TrainConfig
from lora_core_v02 import FluxLoraCoreV02
from get_dataloader_v02 import get_dataloader_v02

# Окончательное ослепление ROCm-драйвера в утилитах diffusers
torch.version.hip, torch.version.rocm = None, None
du.is_rocm_available = lambda: False
du.is_torch_rocm_available = lambda: False

# Автоматический селектор разрядности оптимизатора
try:
    from ao_optim_monolith_v02 import AdamW8bit
    _выбранный_оптимизатор, _режим_8bit = AdamW8bit, True
except ImportError:
    _выбранный_оптимизатор, _режим_8bit = AdamW, False

def init_components():
    """Развертывание ядра Chroma и тотальная очистка видеопамяти."""
    print("[ДВИЖОК] Запуск инициализации маршевых контуров...")
    
    # Подключаем наш новый автономный источник тензоров и ядро LoRA
    _поток_данных = get_dataloader_v02()
    _ядро_модели = FluxLoraCoreV02.init_transformer_with_lora()
    
    # [МАНЕВР АМПУТАЦИИ]: Вышвыриваем тяжелый кодер T5XXL до старта графа оптимизатора
    if hasattr(_ядро_модели, "text_encoder"):
        print("\n***** ВАКУУМНАЯ ВЫГРУЗКА T5XXL: ВЫСВОБОЖДЕНИЕ VRAM *****")
        del _ядро_модели.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    _обучаемые_веса = [п for p in _ядро_модели.parameters() if p.requires_grad]
    _магистральный_оптимизатор = _выбранный_оптимизатор(_обучаемые_веса, lr=TrainConfig.LEARNING_RATE)

    # Принудительное выравнивание матриц для CUDA-ядер
    for _параметр in _обучаемые_веса:
        if _параметр.ndim == 2: 
            _параметр.data = _параметр.data.contiguous()
            
    print("[УСПЕХ] Текстовый кодер аннигилирован, оптимизатор готов к плавке.")
    return _поток_данных, _ядро_модели, _магистральный_оптимизатор
# === КОНЕЦ СЛУЖЕБНОГО КУСКА №2 ===
# === НАЧАЛО СЛУЖЕБНОГО КУСКА №3: МАРШЕВЫЕ ЦИКЛЫ И ВРЕМЕННОЙ ИНЖЕКТОР ===
def run_main_loop(dataloader, model, optimizer):
    """Главный маршевый контур плавки мантиссы Хромы с защитой от WDDM."""
    import time, torch
    from config import TrainConfig
    from step_executor_v02 import execute_single_frame_step
    from telemetry_logger import FluxTelemetryTracker

    _планировщик = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS), eta_min=1e-6)
    _глобальный_шаг = 0
    _бортовой_логгер = FluxTelemetryTracker()

    for _эпоха in range(1, TrainConfig.NUM_EPOCHS + 1):
        model.train()
        for _мега_батч in dataloader:
            _всего_кадров = _мега_батч["latents"].shape[0]
            for _индекс_кадра in range(_всего_кадров):
                _глобальный_шаг += 1
                _лосс_кадра, _шум_cpu, _прогноз_cpu, _таргет_cpu = execute_single_frame_step(
                    mega_batch=_мега_батч, frame_idx=_индекс_кадра, device=torch.device("cuda"), lora_model=model)
                _бортовой_логгер.accumulate_step(_шум_cpu, _прогноз_cpu, _таргет_cpu, _лосс_кадра.item())
# === КОНЕЦ СЛУЖЕБНОГО КУСКА №3 ===
                # === НАЧАЛО СЛУЖЕБНОГО КУСКА №4А: ОПТИМИЗАЦИЯ И МЕТРИКИ ===
                # 3. Жесткий скалярный клиппинг градиентов матриц LoRA (канон 1.0)
                _обучаемые_веса = [п for п in model.parameters() if п.requires_grad]
                torch.nn.utils.clip_grad_norm_(_обучаемые_веса, max_norm=1.0)
                
                # Предохранитель Метрополии: аварийная защита от взрыва градиентов (NaN/Inf)
                if not all(torch.isfinite(_параметр.grad).all() for _параметр in _обучаемые_веса if _параметр.grad is not None):
                    print(f"🚨 [РЕАКТОР] Обнаружены субнормальные числа на шаге {_глобальный_шаг}! Пропуск.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Фиксация приращений LoRA-весов в базовое FP8 тело
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Выжигание мертвого кэша аллокатора CUDA для защиты от WDDM-оверсвапа
                torch.cuda.empty_cache()
                gc.collect()

                # 4. КОСМОФЛОТСКАЯ МГНОВЕННАЯ ТЕЛЕМЕТРИЯ КАДРА
                with torch.no_grad():
                    _мгновенный_лосс = _лосс_кадра.item() * TrainConfig.GRADIENT_ACCUMULATION_STEPS
                    _активная_память = torch.cuda.memory_allocated() / (1024 ** 3)
                    _кэш_аллокатора = torch.cuda.memory_reserved() / (1024 ** 3)
                    
                    print(f"📊 [МАРШ] Шаг: {_глобальный_шаг} | Эпоха: {_эпоха} | Кадр: {_индекс_кадра}")
                    print(f" └── Loss кадра: {_мгновенный_лосс:.4f} | VRAM Активная: {_активная_память:.2f} GB")
                    print(f" └── Кэш CUDA: {_кэш_аллокатора:.2f} GB | Статус: КОНТУР СТАБИЛЕН")
                    print("─" * 60)
                # Передаем управление в Кусок 4Б (Чекпоинты, закрытие циклов и __main__)
                # === КОНЕЦ СЛУЖЕБНОГО КУСКА №4А ===
