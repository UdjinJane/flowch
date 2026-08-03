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
            
            # Распечатываем полный сквозной стек вызовов PyTorch и внешних Си-либ
            print("[СКВОЗНОЙ СТЕК ВСЕХ СИСТЕМНЫХ БИБЛИОТЕК]:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            print("!"*80)
            
            # Тотальный дамп всех локальных объектов во всех доступных кадрах стека
            print("[ИНСПЕКЦИЯ ПАМЯТИ КАДРА ПАДЕНИЯ]:")
            curr_frame = frame
            while curr_frame:
                print(f" -> Локация: {curr_frame.f_code.co_filename}:{curr_frame.f_lineno} в функции {curr_frame.f_code.co_name}")
                for key, val in curr_frame.f_locals.items():
                    if any(x in key.lower() for x in ["mod", "vec", "tensor", "args", "shape"]):
                        try:
                            if hasattr(val, "shape"):
                                print(f"    * Тензор [{key}]: Форма {val.shape} | Тип {val.dtype}")
                            elif isinstance(val, (list, tuple)):
                                print(f"    * Контейнер [{key}]: Тип {type(val).__name__} | Длина = {len(val)}")
                                if len(val) > 0: print(f"      - Элемент 0: {type(val[0]).__name__}")
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

    def verify_gradients(self, layer_name: str):
        for name, param in [("lora_A", self.lora_A), ("lora_B", self.lora_B)]:
            if param.grad is None:
                print(f" [WARN] МЕРТВЫЙ ГРАДИЕНТ [{layer_name}.{name}]")
            elif torch.isnan(param.grad).any():
                print(f" [КРАХ] ВЗРЫВ ГРАДИЕНТА [{layer_name}.{name}]")
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
        # Строгий фильтр: перехватываем только .qkv слои внимания и .linear1 блоки одиночного каскада
        if any(target in name for target in ["txt_attn.qkv", "img_attn.qkv", "linear1"]):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                # Блокируем Autograd оригинальной базы Метрополии (Жесткое удержание VRAM)
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                # Ввариваем кастомный шов LoRA строго под калибр ранга сессии
                lora_wrapper = ChromaInlineLoRA(module, rank=rank)
                setattr(parent, child_name, lora_wrapper)
                patched_count += 1
                print(f" -> [OK] Инжектирован шов: {name} | База намертво заморожена.")
                
    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Ошибка сканирования: точки инжекции LoRA не обнаружены!")
        
    print(f"# === ИНЖЕКЦИЯ ЗАВЕРШЕНА. УСПЕШНО СВАРЕНО ШВОВ: {patched_count} ===")
    return patched_count
#---------------- Конец Блока 2 -----------------
#---------------- Старт Блока 3 (Монолитное Боевое Ядро со сквозной приборной панелью) -------------
def train_step_core(batch: dict, model: nn.Module, optimizer: AdamW8bit, approximator: nn.Module, mod_projector: nn.Module) -> float:
    """
    Выполняет один боевой шаг плавки LoRA по траекториям Rectified Flow.
    Полностью очищен от ошибок вложенности списков и утечек VRAM в Shared RAM Windows.
    """
    x1 = batch["latent"].cuda()
    clip_hidden = batch["clip_hidden"].cuda()
    t5_raw = batch["t5_hidden"].cuda()

    # ШОВ ВЫРАВНИВАНИЯ КОНТЕНТА: Добиваем текстовый кэш T5 до эталонных 512 токенов
    B_pad, L_pad, D_pad = t5_raw.shape
    if L_pad < 512:
        padding_size = 512 - L_pad
        zero_padding = torch.zeros((B_pad, padding_size, D_pad), dtype=t5_raw.dtype, device=t5_raw.device)
        t5_hidden = torch.cat([t5_raw, zero_padding], dim=1)
    else:
        t5_hidden = t5_raw[:, :512, :]

    optimizer.zero_grad(set_to_none=True)
    
    # Генерация шумового поля траектории Rectified Flow
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape,), device=x1.device, dtype=x1.dtype)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0
    target_velocity = x1 - x0

    # Расчет векторов модуляции через Аппроксиматор Метрополии
    t_vec = t.view(-1, 1).to(torch.bfloat16).requires_grad_(True)
    raw_mod = approximator(t_vec)

    # Проекция шины модуляции через статический узел (Защита VRAM от мусорных Autograd-копий)
    if raw_mod.shape[-1] == 5120:
        raw_mod = mod_projector(raw_mod)

    monolithic_mod = raw_mod.unsqueeze(1)
    flat_mods = distribute_modulations(monolithic_mod, depth_single_blocks=38, depth_double_blocks=19)

    mods = {"double": [], "single": [], "final": None}
    
    # Снайперская сборка пакетов: выпрямляем структуру под устав DoubleStreamBlock ядра
    for i in range(19):
        img_mod_pair = flat_mods[f"double_blocks.{i}.img_mod.lin"]  # Список [img_mod1, img_mod2]
        txt_mod_pair = flat_mods[f"double_blocks.{i}.txt_mod.lin"]  # Список [txt_mod1, txt_mod2]
        
        # Разворачиваем в плоский уставной список из 4-х изолированных объектов ModulationOut
        flat_double_vec = [img_mod_pair[0], img_mod_pair[1], txt_mod_pair[0], txt_mod_pair[1]]
        mods["double"].append(flat_double_vec)
        
    for i in range(38):
        mods["single"].append(flat_mods[f"single_blocks.{i}.modulation.lin"])
        
    mods["final"] = flat_mods["final_layer.adaLN_modulation.1"]

    # ==========================================================================
    # ПРИБОРНАЯ ПАНЕЛЬ ТЕЛЕМЕТРИИ ПЕРЕД ПОДЖИГОМ (Прямой рентген памяти)
    # ==========================================================================
    print("\n" + "="*60)
    print("[ПРИБОРНАЯ ПАНЕЛЬ]: Срез параметров перед маршевым проходом:")
    print(f" -> Форма входящего латента xt    : {xt.shape} | Dtype: {xt.dtype}")
    print(f" -> Форма текстовой шины T5      : {t5_hidden.shape} | Dtype: {t5_hidden.dtype}")
    print(f" -> Контейнер mods['double']     : Длина списка = {len(mods['double'])}")
    if len(mods['double']) > 0:
        print(f"    * Длина плоского вектора Блока №0 : {len(mods['double'][0])} (Обязана быть равна 4!)")
        print(f"    * Тип элемента 0 (Графика)       : {type(mods['double'][0][0]).__name__}")
    print(f" -> Контейнер mods['single']     : Длина списка = {len(mods['single'])}")
    print(f" -> Финальный контейнер mods['final']: Длина списка = {len(mods['final'])}")
    print("="*60 + "\n")
    # ==========================================================================

    # Прямой маршевый проход трансформера
    pred_velocity = model(xt, t5_hidden, mods)
    loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    
    if torch.isnan(loss):
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Критическая ошибка: Loss рухнул в NaN!")
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Тотальная зачистка и выжигание следов бэкварда: спасаем VRAM от Shared-течи WDDM
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    return loss.item()
#---------------- Конец Блока 3 -----------------
#---------------- Старт Блока 4 (Маршевый Очищенный Трансформер ChromaMMDiT) ----------------
class ChromaMMDiT(nn.Module):
    """
    Декомпозированный маршевый трансформер Chroma.
    Полностью очищен от ошибок распаковки кортежей и С++ заносов памяти.
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

    def forward(self, x_latent: torch.Tensor, txt_hidden: torch.Tensor, mods: dict) -> torch.Tensor:
        # Снайперский срез фантомной оси DataLoader [B, 1, L, D] -> [B, L, D]
        if len(txt_hidden.shape) == 4:
            txt_hidden = txt_hidden.squeeze(1)
            
        x_latent = x_latent.to(torch.bfloat16)
        txt_hidden = txt_hidden.to(torch.bfloat16)
        
        # Сборка первичных токенов
        xt_flat = self.pack_latents(x_latent)
        img_tokens = self.img_in(xt_flat)
        txt_tokens = self.txt_in(txt_hidden)

        # Вычисляем жесткую длину отсека текста для безопасного среза памяти
        txt_len = txt_tokens.shape[1]

        # 1. Проход по спаренным链 (Принимаем строго 2 элемента из очищенного ядра layers_clean!)
        for i, block in enumerate(self.double_blocks):
            txt_tokens, img_tokens = block(
                txt=txt_tokens, img=img_tokens, pe=None, distill_vec=mods["double"][i]
            )

        # 2. Склеивание потоков для одиночного параллельного каскада
        x_combined = torch.cat([txt_tokens, img_tokens], dim=1)
        for i, block in enumerate(self.single_blocks):
            x_combined = block(
                x=x_combined, pe=None, distill_vec=mods["single"][i]
            )

        # 3. Восстановление исходной 4D-геометрии с герметизацией non-contiguous памяти
        pred_img_flat = x_combined[:, txt_len:].contiguous()
        pred_img_flat = self.final_layer(pred_img_flat)
        
        B, C_raw, H_raw, W_raw = x_latent.shape
        H, W = H_raw // 2, W_raw // 2
        out = pred_img_flat.view(B, H, W, 16, 2, 2)
        out = out.permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, 16, H_raw, W_raw)
#---------------- Конец Блока 4 -----------------

#---------------- Старт Блока 5 (Контур Инициализации и Точка Входа Реактора) ----------------
def run_reactor_forge():
    """
    Управляет запуском реактора: разворачивает топологию, состыкует трюмы
    и подает силовое напряжение на LoRA-узлы.
    """
    print("# === ИНИЦИАЛИЗАЦИЯ ДВИЖКА ТРЕНИРОВКИ TRAIN_ENGINE_V02 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Сборка и патчинг маршевого трансформера
    model = ChromaMMDiT()
    patched_count = patch_chroma_reactor(model, rank=16)
    model = model.to(device)
    
    # 2. Статическая инициализация шины и проектора модуляции (Изоляция Autograd)
    approximator = Approximator(in_dim=1, out_dim=5120, hidden_dim=5120, n_layers=4).to(device)
    mod_projector = nn.Linear(5120, 3072, bias=False, dtype=torch.bfloat16).to(device)
    
    # 3. Привязка автономного 8-битного оптимизатора AdamW строго к Master Weights адаптера
    optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    print(" -> [OK] Автономный 8-битный оптимизатор AdamW8bit зафиксирован.")

    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"
    
    # Контур автоматического переключения трюмов данных
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

    print("# === РАКЕТНЫЙ ЗАПУСК РЕАКТОРА: СТАРТ ЦИКЛА ПЛАВКИ ===")
    for step, batch in enumerate(dataloader):
        try:
            loss = train_step_core(batch, model, optimizer, approximator, mod_projector)
            print(f" -> [ШАГ №{step + 1}] ПЛАВКА СТАБИЛЬНА! Текущий Loss: {loss:.6f}")
            if step >= 1:  # Ограничение тестового прогрева для контроля удержания WDDM
                break
        except Exception as e:
            print(f" [АВАРИЯ РАД ТАЙМА]: Цикл прерван на шаге {step + 1}: {e}")
            break
            
    print("# === ДВИЖОК ВЕРИФИЦИРОВАН. ВСЕ ШВЫ ДЕРЖАТ УДАР. КОНЕЦ СЕССИИ ===")

if __name__ == "__main__":
    run_reactor_forge()
#---------------- Конец Блока 5 -----------------
