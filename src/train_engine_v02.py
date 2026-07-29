#---------- Старт блока №1
import os
import sys
import gc

# Динамическая навигация по полкам проекта
file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(file_path)
project_root = os.path.dirname(src_dir)

# Заталкиваем наши отсеки в самый upper системного поиска
for address in [src_dir, project_root]:
    if address not in sys.path:
        sys.path.insert(0, address)
#--------- Окончание блока №1
#---------- Старт блока №2
import chroma_math
sys.modules["src.math"] = chroma_math

os.environ.update({
    "QUANTO_DISABLE_CPP_EXT": "1",
    "HF_DISABLE_COMPILING": "1",
    "FORCE_CUDA": "1",
    "USE_ROCM": "0"
})

for variable in ["ROCM_HOME", "HIP_PATH", "OLLAMA_LLM_LIBRARY", "ROCM_PATH"]:
    os.environ.pop(variable, None)
#--------- Окончание блока №2
#---------- Старт блока №3
import torch
import diffusers.utils.import_utils as du
from torch.optim import AdamW
from config import TrainConfig
from lora_core_v02 import FluxLoraCoreV02
from get_dataloader_v02 import get_dataloader_v02

torch.version.hip, torch.version.rocm = None, None
du.is_rocm_available = lambda: False
du.is_torch_rocm_available = lambda: False

try:
    from ao_optim_monolith_v02 import AdamW8bit
    chosen_optimizer, mode_8bit = AdamW8bit, True
except ImportError:
    chosen_optimizer, mode_8bit = AdamW, False
#--------- Окончание блока №3
#---------- Старт блока №4
def init_components():
    """Развертывание ядра Chroma и тотальная очистка видеопамяти."""
    print("[ДВИЖОК] Запуск инициализации маршевых контуров...")
    
    data_stream = get_dataloader_v02()
    model_core = FluxLoraCoreV02.init_transformer_with_lora()
    
    if hasattr(model_core, "text_encoder"):
        print("\n***** ВАКУУМНАЯ ВЫГРУЗКА T5XXL: ВЫСВОБОЖДЕНИЕ VRAM *****")
        del model_core.text_encoder
        gc.collect()
        torch.cuda.empty_cache()
#--------- Окончание блока №4
#---------- Старт блока №5
    trainable_weights = [p for p in model_core.parameters() if p.requires_grad]
    backbone_optimizer = chosen_optimizer(trainable_weights, lr=TrainConfig.LEARNING_RATE)
    
    for parameter in trainable_weights:
        if parameter.ndim == 2:
            parameter.data = parameter.data.contiguous()
            
    print("[УСПЕХ] Текстовый кодер аннигилирован, оптимизатор готов к плавке.")
    return data_stream, model_core, backbone_optimizer
#--------- Окончание блока №5
#---------- Старт блока №6
def run_main_loop(dataloader, model, optimizer):
    """Главный маршевый контур плавки мантиссы Хромы с защитой от WDDM."""
    import time
    import torch
    from config import TrainConfig
    from step_executor_v02 import execute_single_frame_step
    from telemetry_logger import FluxTelemetryTracker
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=getattr(TrainConfig, "TOTAL_STEPS", TrainConfig.MAX_TRAIN_STEPS), 
        eta_min=1e-6
    )
    global_step = 0
    board_logger = FluxTelemetryTracker()
#--------- Окончание блока №6
#---------- Старт блока №7
    for epoch in range(1, TrainConfig.NUM_EPOCHS + 1):
        model.train()
        for mega_batch in dataloader:
            total_frames = mega_batch["latents"].shape[0]
            for frame_idx in range(total_frames):
                global_step += 1
                
                frame_loss, noise_cpu, prediction_cpu, target_cpu = execute_single_frame_step(
                    mega_batch=mega_batch, 
                    frame_idx=frame_idx, 
                    device=torch.device("cuda"), 
                    lora_model=model
                )
                board_logger.accumulate_step(noise_cpu, prediction_cpu, target_cpu, frame_loss.item())
#--------- Окончание блока №7
#---------- Старт блока №8
                from step_optimizer_v02 import run_optimizer_and_telemetry
                step_success = run_optimizer_and_telemetry(
                    model, optimizer, frame_loss, global_step, epoch, frame_idx
                )
                if not step_success:
                    continue

if __name__ == "__main__":
    dataloader, model, optimizer = init_components()
    run_main_loop(dataloader, model, optimizer)
#--------- Окончание блока №8
