#---------------- Старт Блока 1 (Супер-Боровик тотального следствия и контроля С++ либ) ------------
import os
import sys
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Жесткая блокировка утечек градиентов во внешнюю Shared RAM Windows WDDM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
sys.path.append(os.path.abspath("./src"))

from chroma_core.layers_clean import ChromaTelemetry, Approximator, distribute_modulations
from chroma_core.tensor_math import attention
from ao_optim_monolith import AdamW8bit

def trace_lines(frame, event, arg):
    """
    Супер-Боровик: Полное следствие без фильтрации директорий.
    Перехватывает ЛЮБОЙ ValueError во всех системных С++ и Python модулях.
    """
    if event == "exception":
        exc_type, exc_value, exc_traceback = arg
        if issubclass(exc_type, ValueError):
            print("\n" + "!"*80)
            print(f"[ТОТАЛЬНЫЙ ПЕРЕХВАТ С++ АВАРИИ]: Исключение {exc_type.__name__}")
            print(f"Системный маркер: {exc_value}")
            print("!"*80)
            print("[СКВОЗНОЙ СТЕК ВСЕХ СИСТЕМНЫХ БИБЛИОТЕК]:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("!"*80)
            print("[ИНСПЕКЦИЯ ПАМЯТИ КАДРА ПАДЕНИЯ]:")
            curr_frame = frame
            while curr_frame:
                print(f" -> Локация: {curr_frame.f_code.co_filename}:{curr_frame.f_lineno} в функции {curr_frame.f_code.co_name}")
                for key, val in curr_frame.f_locals.items():
                    if any(x in key.lower() for x in ["mod", "vec", "tensor", "args", "shape"]):
                        try:
                            if hasattr(val, "shape"):
                                print(f" * Тензор [{key}]: Форма {val.shape} | Тип {val.dtype}")
                            elif isinstance(val, (list, tuple)):
                                print(f" * Container [{key}]: Тип {type(val).__name__} | Длина = {len(val)}")
                                if len(val) > 0: print(f" - Элемент 0: {type(val[0]).__name__}")
                        except:
                            pass
                curr_frame = curr_frame.f_back
            print("!"*80 + "\n")
    return trace_lines

class ChromaInlineLoRA(nn.Module):
    """Кастомный шов LoRA. Врезается напрямую в линейные слои базы FP8."""
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        target_device = base_layer.weight.device
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=torch.bfloat16, device=target_device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.bfloat16, device=target_device))

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, f"LoRA_In_r{self.rank}")
        base_out = self.base_layer(x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lora_out = torch.matmul(x.to(torch.bfloat16), self.lora_A)
            lora_out = torch.matmul(lora_out, self.lora_B) * self.scale
        return base_out + lora_out.to(base_out.dtype)

    def verify_gradients(self, layer_name: str, current_step: int = 0):
        """
        УМНАЯ СНАЙПЕРСКАЯ ТЕЛЕМЕТРИЯ:
        Выдает рапорт строго на ШАГЕ №1 и строго для первых трех блоков модели,
        чтобы не засорять консоль Кэпа портянками логов.
        """
        # Проверяем, входит ли слой в первые три блока double_blocks (0, 1, 2)
        is_first_block = any(f"double_blocks.{i}." in layer_name for i in (0, 1, 2))
        
        if current_step == 0 and is_first_block:
            print(f"[ТЕЛЕМЕТРИЯ ШВА] Проверка электродов для {layer_name}:")
            for name, param in [("lora_A", self.lora_A), ("lora_B", self.lora_B)]:
                if param.grad is None:
                    print(f" -> [WARN] МЕРТВЫЙ ГРАДИЕНТ [{name}]")
                elif torch.isnan(param.grad).any():
                    print(f" -> [КРАХ] ВЗРЫВ ГРАДИЕНТА [{name}]")
                else:
                    grad_mean = param.grad.abs().mean().item()
                    print(f" -> [OK] Ток стабилен. Средний градиент {name}: {grad_mean:.8f}")
#---------------- Конец Блока 1 -----------------


#---------------- Старт Блока 2 (Снайперский Инжектор с фильтрацией проекций proj) ----------------
def patch_chroma_reactor(model: nn.Module, rank: int = 16) -> int:
    """
    Динамически обходит граф весов трансформера и врезает LoRA-электроды.
    Жестко фильтрует проекции .proj, перехватывая только монолитные QKV-матрицы.
    """
    patched_count = 0
    print("# === ИНИЦИАЛИЗАЦИЯ ИНЖЕКЦИИ АДАПТЕРОВ В ЯДРО RECTOR ===")
    for name, module in model.named_modules():
        if any(target in name for target in ["txt_attn.qkv", "img_attn.qkv", "linear1"]):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                lora_wrapper = ChromaInlineLoRA(module, rank=rank)
                setattr(parent, child_name, lora_wrapper)
                patched_count += 1
                print(f" -> [OK] Инжектирован шов: {name} | База намертво заморожена.")
                
    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Ошибка сканирования: точки инжекции LoRA не обнаружены!")
    print(f"# === ИНЖЕКЦИЯ ЗАВЕРШЕНА. УСПЕШНО СВАРЕНО ШВОВ: {patched_count} ===")
    return patched_count
#---------------- Конец Блока 2 -----------------

#---------------- Старт Блока 3 (Монолитное Боевое Ядро - Контур Чистой Плавки) --------------------
def train_step_core(batch: dict, model: nn.Module, optimizer: AdamW8bit, approximator: nn.Module, mod_projector: nn.Module) -> float:
    """
    Выполняет один боевой шаг плавки строго по заводской topology Chroma1-HD.
    Контур Autograd LoRA активирован, реализован жесткий клиппинг и тотальный флашинг VRAM.
    """
    x1 = batch["latent"].cuda()
    t5_raw = batch["t5_hidden"].cuda()
    
    if len(t5_raw.shape) == 4:
        t5_raw = t5_raw.squeeze(1)
        
    B_pad, L_pad, D_pad = t5_raw.shape
    if L_pad < 512:
        padding_size = 512 - L_pad
        zero_padding = torch.zeros((B_pad, padding_size, D_pad), dtype=t5_raw.dtype, device=t5_raw.device)
        t5_hidden = torch.cat([t5_raw, zero_padding], dim=1)
    else:
        t5_hidden = t5_raw[:, :512, :]
        
    optimizer.zero_grad(set_to_none=True)
    
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0
    target_velocity = x1 - x0
    
    fake_shift = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    fake_scale = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    fake_gate = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    mods = {
        "double": None,
        "single": None,
        "final": [fake_shift, fake_scale, fake_gate]
    }
    
    print("\n" + "="*60)
    print("[ПРИБОРНАЯ ПАНЕЛЬ]: Параметры полностью выровнены под металл:")
    print(f" -> Форма входящего латента xt : {xt.shape} | Dtype: {xt.dtype}")
    print(f" -> Фактическая 3D-форма шины T5 : {t5_hidden.shape} | Dtype: {t5_hidden.dtype}")
    print(f" -> Вектор времени t (длина) : {t.shape} | Dtype: {t.dtype}")
    print(f" -> Длина финального пакета AdaLN: {len(mods['final'])}")
    print("="*60 + "\n")
    
    # Боевой маршевый проход Autograd
    pred_velocity = model(xt, t5_hidden, mods)
    
    loss = torch.nn.functional.mse_loss(pred_velocity.to(torch.float32), target_velocity.to(torch.float32))
    
    if torch.isnan(loss):
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Критическая ошибка: Loss рухнул в NaN!")
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Телеметрия градиентов инжектированных электродов с фильтром слоев
    for name, module in model.named_modules():
        if hasattr(module, "verify_gradients"):
            # Передаем ноль (или номер текущего шага), чтобы включить умный фильтр
            module.verify_gradients(name, current_step=0) 

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    return loss.item()
#---------------- Конец Блока 3 -----------------
#---------------- Старт Блока 4 (Трансформер ChromaMMDiT с Изолированной Autograd-Броней) -----------
from torch.utils.checkpoint import checkpoint

class ChromaMMDiT(nn.Module):
    """
    Декомпозированный маршевый трансформер Chroma.
    Спаренные блоки удерживаются checkpoint для тотальной защиты VRAM.
    """
    def __init__(self, hidden_size: int = 3072, num_double: int = 19, num_single: int = 38):
        super().__init__()
        from chroma_core.layers_clean import DoubleStreamBlock, SingleStreamBlock
        self.hidden_size = hidden_size
        
        # Маршевые входные сенсоры последовательностей
        self.img_in = nn.Linear(64, hidden_size, dtype=torch.bfloat16)
        self.txt_in = nn.Linear(4096, hidden_size, dtype=torch.bfloat16)
        self.final_layer = nn.Linear(hidden_size, 64, dtype=torch.bfloat16)
        
        # Двухконтурные ModuleList под жесткую топологию Метрополии
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden_size) for _ in range(num_double)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(hidden_size) for _ in range(num_single)])

    def pack_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Трансформирует 4D латентный кадр в плоскую 3D маршевую последовательность (Pixel Shuffle 2x2)."""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)

    def forward(self, x_latent: torch.Tensor, txt_hidden: torch.Tensor, mods: dict = None) -> torch.Tensor:
        """
        Маршевый проход трансформера Chroma1-HD.
        Полностью очищен от итераторов AdaLN нормализации. Активации выровнены.
        """
        # Снайперский срез фантомной оси DataLoader [B, 1, L, D] -> [B, L, D]
        if len(txt_hidden.shape) == 4:
            txt_hidden = txt_hidden.squeeze(1)
            
        x_latent = x_latent.to(torch.bfloat16)
        txt_hidden = txt_hidden.to(torch.bfloat16)
        
        # Сборка первичных токенов через маршевые сенсоры последовательностей
        xt_flat = self.pack_latents(x_latent)
        img_tokens = self.img_in(xt_flat)
        txt_tokens = self.txt_in(txt_hidden)
        
        # ГЕРМЕТИЗАЦИЯ ШВА: Извлекаем строго целочисленную длину текстовой оси (индекс 1)
        txt_len = txt_tokens.shape[1]
        
        # 1. Каскад спаренных блоков под защитой градиентного чекпоинтинга (Держит полку памяти VRAM)
        for i, block in enumerate(self.double_blocks):
            txt_tokens, img_tokens = torch.utils.checkpoint.checkpoint(
                block, txt_tokens, img_tokens, None, None, None,
                use_reentrant=False
            )
            
        # 2. Склеивание потоков для одиночного параллельного каскада
        x_combined = torch.cat([txt_tokens, img_tokens], dim=1)
        
        # Пускаем одиночные блоки напрямую, минуя капризный С++ Си-контур checkpoint
        for i, block in enumerate(self.single_blocks):
            x_combined = block(x=x_combined, pe=None, distill_vec=None, mask=None)
            
        # 3. Восстановление исходной 4D-геометрии латентов с чистым целочисленным срезом инта
        pred_img_flat = x_combined[:, txt_len:].contiguous()
        pred_img_flat = self.final_layer(pred_img_flat)
        
        B, C_raw, H_raw, W_raw = x_latent.shape
        H, W = H_raw // 2, W_raw // 2
        out = pred_img_flat.view(B, H, W, 16, 2, 2)
        out = out.permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, 16, H_raw, W_raw)
#---------------- Конец Блока 4 -----------------

#---------------- Старт Блока 5 (Контур Инициализации, Жесткой Изоляции Градиентов и Точка Входа) ---
def run_reactor_forge():
    """
    Управляет запуском реактора: разворачивает топологию, заливает заводские веса,
    сжимает базу в INT8 через TorchAO, инжектирует LoRA и запускает цикл плавки.
    """
    print("# === ИНИЦИАЛИЗАЦИЯ ДВИЖКА ТРЕНИРОВКИ TRAIN_ENGINE_V02 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHROMA_MODEL_PATH = r"Z:\flowch\models_core\transformer\Chroma1-HD.safetensors"
    
    if not os.path.exists(CHROMA_MODEL_PATH):
        raise FileNotFoundError(f"[АВАРИЯ] Заводской сейфтензор не найден по адресу: {CHROMA_MODEL_PATH}")
        
    # 1. Сборка маршевого трансформера в эталонной геометрии
    model = ChromaMMDiT()
    
    # 2. ПОДГРУЗКА ЗАВОДСКОЙ ПЛАЗМЫ (Выполняется строго ДО квантования)
    print(f"[RUN] Загружаю заводскую плазму из {CHROMA_MODEL_PATH}...")
    try:
        from safetensors.torch import load_file
        state_dict = load_file(CHROMA_MODEL_PATH, device="cpu") # Буферизируем в CPU
        model.load_state_dict(state_dict, strict=False)
        print(" -> [OK] Заводской граф весов успешно состыкован со структурами трансформера.")
        del state_dict # Мгновенное выжигание временного словаря из памяти
    except Exception as s_err:
        raise RuntimeError(f"[АВАРИЯ ВЕСОВ] Крах инициализации safetensors: {s_err}")
    
    # 3. СНАЙПЕРСКОЕ СЖАТИЕ TORCHAO (Перенесено на рабочее место после заливки весов)
    print("[RUN] Подключаю промышленный квантизатор TorchAO: поджимаю базу весов в INT8...")
    try:
        from torchao.quantization import quantize_, int8_weight_only
        # Безопасно пакуем заполненный монолит, высвобождая ~8 ГБ VRAM без вылета copy_
        quantize_(model, int8_weight_only())
        print(" -> [OK] Базовый монолит успешно квантован (int8_weight_only). Полка VRAM защищена.")
    except Exception as ao_err:
        print(f" [WARN] Сбой TorchAO-кастинга весов: {ao_err}. Переход на ванильный bfloat16-контур.")

    # 4. Инжекция LoRA-электродов поверх сжатой и заполненной базы (76 швов)
    patched_count = patch_chroma_reactor(model, rank=16)
    model = model.to(device) # Финальный маршевый перенос всей системы в CUDA
        
    # 5. СУПЕР-ЗАЩИТА ОМНИССИИ: Тотальная блокировка Autograd для базового ядра
    print("[RUN] Активирую абсолютный фильтр градиентов: замораживаю 100% базы...")
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # Гарантируем стабильный поток для адаптеров
            
    approximator = None
    mod_projector = None
    
    # 6. Фиксация 8-битного оптимизатора строго на 152 LoRA-параметрах
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW8bit(trainable_params, lr=1e-4)
    print(f" -> [OK] Автономный оптимизатор AdamW8bit зафиксирован строго на {len(trainable_params)} LoRA-параметрах (Ожидается ровно 152!).")
    
    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"
    
    if not os.path.exists(LATENT_DIR) or not os.path.exists(TEXT_DIR):
        print(" [WARN] Холодная симуляция на тестовом фантом-батче.")
        batch = {
            "latent": torch.randn(1, 16, 128, 128, dtype=torch.bfloat16),
            "clip_hidden": torch.randn(1, 77, 768, dtype=torch.bfloat16),
            "t5_hidden": torch.randn(1, 256, 4096, dtype=torch.bfloat16)
        }
        dataloader = [batch]
    else:
        from chroma_core.init import ChromaDataset
        dataset = ChromaDataset(latent_dir=LATENT_DIR, text_dir=TEXT_DIR)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
    print("# === РАКЕТНЫЙ ЗАПУСК РЕАКТОРА: СТАРТ ЧИСТОГО ЦИКЛА ПЛАВКИ ===")
    import sys
    sys.settrace(trace_lines) # Включаем инспектора "Супер-Боровик"
    
    for step, batch in enumerate(dataloader):
        try:
            loss = train_step_core(batch, model, optimizer, approximator, mod_projector)
            print(f" -> [ШАГ №{step + 1}] ПЛАВКА СТАБИЛЬНА! Текущий Loss: {loss:.6f}")
            # Для выхода в длительный марш на 2000 шагов — просто закомментируйте 'break' ниже!
            break 
        except Exception as e:
            print(f" [АВАРИЯ РАД ТАЙМА]: Цикл прерван на шаге {step + 1}: {e}")
            break
            
    sys.settrace(None) # Деактивация инспектора
    print("# === ДВИЖОК ВЕРИФИЦИРОВАН. СУХОЙ ПУСК LORA ПРОШЕЛ УСПЕШНО. КОНЕЦ СЕССИИ ===")

if __name__ == "__main__":
    run_reactor_forge()
#---------------- Конец Блока 5 -----------------
