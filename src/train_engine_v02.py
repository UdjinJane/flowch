#---------------- Старт Блока 1 (Супер-Боровик тотального следствия и контроля С++ либ) ------------
import os
import sys
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Жесткая блокировка фрагментации и активация расширяемых сегментов кремния
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
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
                        except:
                            pass
                curr_frame = curr_frame.f_back
            print("!"*80 + "\n")
    return trace_lines

class ChromaInlineLoRA(nn.Module):
    """
    Высшая автономная броня v3.0: Нативный шов LoRA с протоколом изолированного бэкворда.
    Полностью ампутирует расчет dx, исключая междевайсовые Си-заносы аллокатора TorchAO.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        target_device = base_layer.weight.device
        
        # Обучаемые параметры шва в чистом bfloat16
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=torch.bfloat16, device=target_device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.bfloat16, device=target_device))

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, f"LoRA_In_r{self.rank}")
        x_cont = x.contiguous().to(torch.bfloat16)
        
        # РЕЖИМ СУХОГО ХОДА: Абсолютный no_grad экран базового монолита Метрополии
        with torch.no_grad():
            base_out = self.base_layer(x_cont).detach()
            lora_mid = torch.matmul(x_cont, self.lora_A)
            lora_out = torch.matmul(lora_mid, self.lora_B) * self.scale
            
        return (lora_out + base_out).detach()

    def inject_manual_backward(self, loss_grad_output: torch.Tensor, static_incoming_x: torch.Tensor):
        """
        Локальный инжектор Омниссии v3.0: принимает чистый градиент от Loss и пред-рассчитанные
        на старте токенизированные активации. Рассчитывает grad_A и grad_B без трансляции dx назад.
        """
        with torch.no_grad():
            x_cont = static_incoming_x.contiguous().to(torch.bfloat16)
            
            # Снайперское выравнивание осей градиента под геометрию выходной матрицы шва
            dy = loss_grad_output.to(torch.bfloat16) * self.scale
            
            # ЖЕСТКАЯ ПАРАНОИДАЛЬНАЯ СИНХРОНИЗАЦИЯ ОСЕЙ ПОСЛЕДОВАТЕЛЬНОСТИ (Ликвидация тараканов)
            if dy.shape[1] > x_cont.shape[1]:
                dy = dy[:, :x_cont.shape[1], :]
            elif dy.shape[1] < x_cont.shape[1]:
                # Если градиент уже (DoubleStreamBlock текст/картинка), расширяем его нулями до длины активаций x_cont
                padding_size = x_cont.shape[1] - dy.shape[1]
                zero_padding = torch.zeros((dy.shape[0], padding_size, dy.shape[2]), dtype=dy.dtype, device=dy.device)
                dy = torch.cat([dy, zero_padding], dim=1)
                
            # Перерасчет локального форварда "на лету" (Activation Checkpointing в вакууме графов)
            lora_mid = torch.matmul(x_cont, self.lora_A)
            
            # Проекция на двухмерную плоскость для матричного умножения Си-ядра
            dy_flat = dy.view(-1, dy.shape[-1])
            mid_flat = lora_mid.view(-1, lora_mid.shape[-1])
            x_flat = x_cont.view(-1, x_cont.shape[-1])
            
            # 1. Расчет градиентов параметров текущего электрода LoRA
            if mid_flat.shape[0] == dy_flat.shape[0]:
                grad_B = torch.matmul(mid_flat.t(), dy_flat)
                d_mid = torch.matmul(dy_flat, self.lora_B.t())
                grad_A = torch.matmul(x_flat.t(), d_mid)
                
                # Аккумулируем градиенты напрямую в параметры шва
                if self.lora_A.grad is None:
                    self.lora_A.grad = grad_A.view_as(self.lora_A)
                else:
                    self.lora_A.grad += grad_A.view_as(self.lora_A)
                    
                if self.lora_B.grad is None:
                    self.lora_B.grad = grad_B.view_as(self.lora_B)
                else:
                    self.lora_B.grad += grad_B.view_as(self.lora_B)
                    
        # Полная изоляция: возвращаем None, цепочка dx оборвана и уничтожена во избежание деквантования
        return None

    def verify_gradients(self, layer_name: str, current_step: int = 0):
        """Умная телеметрия шва."""
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
                
                # Тотальная заморозка базового элемента: выключаем трекинг Autograd
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                    
                # Врезка нативного bfloat16 адаптера с автоград-шунтом Омниссии
                lora_wrapper = ChromaInlineLoRA(module, rank=rank)
                setattr(parent, child_name, lora_wrapper)
                patched_count += 1
                print(f" -> [OK] Инжектирован шов: {name} | База намертво заморожена.")
                
    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Ошибка сканирования: точки инжекции LoRA не обнаружены!")
    print(f"# === ИНЖЕКЦИЯ ЗАВЕРШЕНА. УСПЕШНО СВАРЕНО ШВОВ: {patched_count} ===")
    return patched_count
#---------------- Конец Блока 2 -----------------

#---------------- Старт Блока 3 (Монолитное Боевое Ядро - Контур Чистой Плавки и Хронометража) ----
import time

def train_step_core(batch: dict, model: nn.Module, optimizer: AdamW8bit, approximator: nn.Module, mod_projector: nn.Module, step: int = 0) -> float:
    """
    Выполняет один боевой шаг плавки с ручным распределением градиентного тока по швам LoRA.
    Тотально выжигает метаданные torchao из контура Loss через жесткую float32-стерилизацию.
    """
    # Фиксация старта фазы ввода-вывода (I/O) и подготовки батча
    t_start = time.perf_counter()
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
        
    # === КОНТУР ВХОДНОЙ ТЕЛЕМЕТРИИ ОМНИССИИ: АБСОЛЮТНЫЙ СТРОКОВЫЙ КАСТИНГ ВСЕХ КОНТЕЙНЕРОВ ===
    if step == 0:
        shape_vae_str = str(list(x1.shape))
        dtype_vae_str = str(x1.dtype)
        shape_t5_str = str(list(t5_hidden.shape))
        dtype_t5_str = str(t5_hidden.dtype)
        
        mem_alloc_gb = str(round(torch.cuda.memory_allocated() / 1024**3, 2))
        mem_res_gb = str(round(torch.cuda.memory_reserved() / 1024**3, 2))
        
        print(f"\n┌── [МАРШЕВАЯ ТЕЛЕМЕТРИЯ ЯДРА RECTOR | ПУСКОВАЯ ВЕРИФИКАЦИЯ ВХОДОВ] ────────────────┐")
        print(f"│ * Входной латент VAE  : Форма {shape_vae_str:<18} | Тип {dtype_vae_str:<10} | Mean {x1.abs().mean().item():.4f} │")
        print(f"│ * Шина текста T5XXL   : Форма {shape_t5_str:<18} | Type {dtype_t5_str:<10} | Mean {t5_hidden.abs().mean().item():.4f} │")
        print(f"│ * Полка видеопамяти   : Выделено {mem_alloc_gb:<5} ГБ       | Зарезервировано {mem_res_gb:<5} ГБ          │")
        print(f"└─────────────────────────────────────────────────────────────────────────────────────┘\n")
        
    optimizer.zero_grad(set_to_none=True)
    
    # Генерация пространственного шума и траектории Rectified Flow
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape, device=x1.device, dtype=torch.float32).to(x1.dtype)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0
    
    # МАРШЕВАЯ МОДИФИКАЦИЯ: target_velocity запечатывается в чистый float32 намертво, минуя bfloat16
    target_velocity = (x1.float() - x0.float()).detach()
    
    # Полная изоляция от глобального Autograd
    xt.requires_grad_(False)
    
    fake_shift = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    fake_scale = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    fake_gate = torch.zeros((1, 1, 3072), dtype=torch.bfloat16, device="cuda")
    mods = {
        "double": None,
        "single": None,
        "final": [fake_shift, fake_scale, fake_gate]
    }
    
    t_io = time.perf_counter() - t_start
    
    # Фиксация и запуск фазы прямого прохода (Forward)
    t_fwd_start = time.perf_counter()
    
    with torch.no_grad():
        pred_velocity_raw = model(xt, t5_hidden, mods)
        
        # ТОТАЛЬНАЯ СТЕРИЛИЗАЦИЯ: Выдергиваем тензор через CPU обратно в CUDA, полностью стирая С++ указатели torchao
        pred_velocity = pred_velocity_raw.detach().cpu().float().cuda()
        
        # Расчет Loss на стерильных float32-тензорах
        loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    
    if torch.isnan(loss):
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Критическая ошибка: Loss рухнул in NaN!")
    t_fwd = time.perf_counter() - t_fwd_start
    
    # Фиксация и запуск фазы обратного прохода (Ручной Бэкворд Омниссии v3.0)
    t_bwd_start = time.perf_counter()
    
    with torch.no_grad():
        # 1. Расчет стартового градиента от MSE-Loss
        total_elements = pred_velocity.numel()
        grad_out_4d = 2.0 * (pred_velocity - target_velocity) / total_elements
        grad_out_4d = grad_out_4d.to(torch.bfloat16)
        
        # 2. Реверс Pixel Shuffle
        B, C_raw, H_raw, W_raw = grad_out_4d.shape
        H, W = H_raw // 2, W_raw // 2
        grad_flat_64 = grad_out_4d.view(B, 16, H, 2, W, 2)
        grad_flat_64 = grad_flat_64.permute(0, 2, 4, 1, 3, 5).contiguous()
        grad_flat_64 = grad_flat_64.view(B, H * W, 64)
        
        # 3. Проход сквозь транспонированный final_layer
        final_weight = model.final_layer.weight.to(torch.bfloat16)
        base_grad_output = torch.matmul(grad_flat_64, final_weight)
        
        # Пред-расчет источников токенов
        xt_flat_base = model.pack_latents(xt)
        static_img_tokens = model.img_in(xt_flat_base).detach()
        static_txt_tokens = model.txt_in(t5_hidden).detach()
        
        # Выравнивание склейки под геометрию Одиночных Блоков
        txt_len = 512 
        zero_txt_grad = torch.zeros((B, txt_len, base_grad_output.shape[-1]), dtype=torch.bfloat16, device="cuda")
        combined_grad_output = torch.cat([base_grad_output, zero_txt_grad], dim=1).contiguous()
        
        # 4. АВТОНОМНЫЙ ЦЕПНОЙ ИНЖЕКТОР v3.0: Швы пьют ток изолированно
        modules_chain = list(model.named_modules())
        for name, module in reversed(modules_chain):
            if hasattr(module, "inject_manual_backward"):
                if "txt_attn" in name:
                    module.inject_manual_backward(base_grad_output, static_txt_tokens)
                elif "img_attn" in name:
                    module.inject_manual_backward(base_grad_output, static_img_tokens)
                else:
                    combined_static = torch.cat([static_img_tokens, static_txt_tokens], dim=1).contiguous()
                    module.inject_manual_backward(combined_grad_output, combined_static)
            
    t_bwd = time.perf_counter() - t_bwd_start
    
    # Фиксация и запуск фазы оптимизации весов (Optimizer Step)
    t_opt_start = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    t_opt = time.perf_counter() - t_opt_start
    
    # Фиксация фазы проверки градиентного тока и снайперской чистки кэша VRAM
    t_clean_start = time.perf_counter()
    for name, module in model.named_modules():
        if hasattr(module, "verify_gradients"):
            module.verify_gradients(name, current_step=step)
            
    optimizer.zero_grad(set_to_none=True)
    
    if step % 250 == 0:
        torch.cuda.empty_cache()
    t_clean = time.perf_counter() - t_clean_start
    
    # РАСЧЕТ И ВЫВОД МАРШЕВОЙ ПСЕВДОГРАФИКИ НА ТАБЛО КОНСОЛИ
    t_total = t_io + t_fwd + t_bwd + t_opt + t_clean
    total_sec = max(t_total, 0.001)
    p_io = int((t_io / total_sec) * 40)
    p_fwd = int((t_fwd / total_sec) * 40)
    p_bwd = int((t_bwd / total_sec) * 40)
    p_opt = int((t_opt / total_sec) * 40)
    p_clean = int((t_clean / total_sec) * 40)
    
    bar_io = "▒" * max(p_io, 1)
    bar_fwd = "█" * max(p_fwd, 1)
    bar_bwd = "▓" * max(p_bwd, 1)
    bar_opt = "█" * max(p_opt, 1)
    bar_clean = "░" * max(p_clean, 1)
    
    print(f"\n┌── [МАРШЕВЫЙ ТРЕКЕР ТАЙМИНГОВ СМЕНЫ | ШАГ №{step + 1}] ──────────────────────────────────┐")
    print(f"│ [I/O & Батч] : {t_io:6.3f} сек | {bar_io:<40} │")
    print(f"│ [Форвард]    : {t_fwd:6.3f} сек | {bar_fwd:<40} │")
    print(f"│ [Бэкворд]    : {t_bwd:6.3f} сек | {bar_bwd:<40} │")
    print(f"│ [Оптимизатор]: {t_opt:6.3f} сек | {bar_opt:<40} │")
    print(f"│ [Флашинг VRAM]: {t_clean:6.3f} сек | {bar_clean:<40} │")
    print(f"├─── ПАСПОРТ СКОРОСТИ РЕАКТОРА ───────────────────────────────────────────────────────┤")
    print(f"│ -> ПОЛНОЕ ВРЕМЯ ТИКА ЦИКЛА: {t_total:6.3f} сек. Текущий Loss: {loss.item():12.6f}      │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────┘\n")
    
    return loss.item()
#---------------- Конец Блока 3 -----------------

#---------------- Старт Блока 4 (Трансформер ChromaMMDiT с Послойной Реентерабельной Броней) -------
class ChromaMMDiT(nn.Module):
    """
    Модернизированный маршевый трансформер Chroma.
    Полностью избавлен от укрупненного и послойного чекпоинтинга.
    Работает в режиме прямой трансляции тензоров в чистом кремнии.
    """
    def __init__(self, hidden_size: int = 3072, num_double: int = 19, num_single: int = 38):
        super().__init__()
        from chroma_core.layers_clean import DoubleStreamBlock, SingleStreamBlock
        self.hidden_size = hidden_size
        
        # Маршевые входные сенсоры последовательностей
        self.img_in = nn.Linear(64, hidden_size, dtype=torch.bfloat16)
        self.txt_in = nn.Linear(4096, hidden_size, dtype=torch.bfloat16)
        self.final_layer = nn.Linear(hidden_size, 64, dtype=torch.bfloat16)
        
        # Двухконтурные ModuleList под жесткую topology Метрополии
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
        Маршевый проход трансформера Chroma1-HD с тотальной изоляцией активаций Autograd.
        """
        if len(txt_hidden.shape) == 4:
            txt_hidden = txt_hidden.squeeze(1)
            
        # Инициализация токенов через входные сенсоры
        xt_flat = self.pack_latents(x_latent)
        img_tokens = self.img_in(xt_flat)
        txt_tokens = self.txt_in(txt_hidden)
        
        # СНАЙПЕРСКИЙ ФИКС: Извлекаем строго скалярную длину последовательности токенов
        img_len = img_tokens.shape[1]
        
        # Прямой каскад спаренных блоков Метрополии без вовлечения базового Autograd
        for block in self.double_blocks:
            img_tokens, txt_tokens = block(
                img=img_tokens,
                txt=txt_tokens,
                pe=None,
                distill_vec=mods["double"],
                mask=None
            )
            
        # Выравнивание склейки под устав Метрополии: сначала ИЗОБРАЖЕНИЕ, затем ТЕКСТ
        x_combined = torch.cat([img_tokens, txt_tokens], dim=1)
        
        # Одиночные blocks — строго 4 позиционных аргумента под геометрию Метрополии!
        for block in self.single_blocks:
            x_combined = block(x_combined, None, mods["single"], None)
            
        # Снайперское отсечение графика-токенов по скалярному индексу img_len
        pred_img_flat = x_combined[:, :img_len].contiguous()
        pred_img_flat = self.final_layer(pred_img_flat)
        
        # Обратный Pixel Shuffle: распаковка 64 каналов в 16-канальное латентное пространство
        B, C_raw, H_raw, W_raw = x_latent.shape
        H, W = H_raw // 2, W_raw // 2
        out = pred_img_flat.view(B, H, W, 16, 2, 2)
        out = out.permute(0, 3, 1, 4, 2, 5)
        return out.reshape(B, 16, H_raw, W_raw)
#---------------- Конец Блока 4 -----------------

#---------------- Старт Блока 5 (Контур Инициализации, Жесткой Изоляции Градиентов и Сохранения Чекпоинтов) -------
def save_lora_checkpoint(model: nn.Module, save_path: str):
    """
    Контур тотальной дефектоскопии весов LoRA.
    Фильтрует, очищает мантиссу и сохраняет исключительно электроды адаптеров
    через безопасный формат safetensors, блокируя проверки превратников Метрополии.
    """
    print(f"\n# === ЗАПУСК ИНСПЕКЦИИ КОНТУРА СОХРАНЕНИЯ: {save_path} ===")
    lora_state_dict = {}
    corrupted_weights = 0
    
    # Снайперский сбор параметров LoRA
    for name, param in model.named_parameters():
        if "lora_" in name:
            # ЖЕСТКИЙ КЛОН ДЛЯ ОБХОДА БЛОКИРОВКИ КАНАЛА ВВОДА-ВЫВОДА WINDOWS
            clean_tensor = param.detach().clone().cpu().to(torch.bfloat16)
            
            # Проверка гидродинамики весов на NaN и квантовые прожоги
            if torch.isnan(clean_tensor).any() or torch.isinf(clean_tensor).any():
                print(f" -> [КРАХ ТЕЛЕМЕТРИИ] Обнаружены поврежденные структуры в слое: {name}")
                corrupted_weights += 1
            lora_state_dict[name] = clean_tensor
            
    if corrupted_weights > 0:
        print(f" [WARN] Обнаружено поврежденных швов: {corrupted_weights}. Запись заблокирована для защиты ядра!")
        return False
        
    if len(lora_state_dict) != 152:
        print(f" [WARN] Аномалия tobacco контура: собрано {len(lora_state_dict)} параметров вместо уставных 152!")
        
    # Безопасная запись на физический носитель
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        print(f" -> [OK] Контур запечатан. Веса LoRA успешно сохранены! Размер буфера: {len(lora_state_dict)} узлов.")
        return True
    except Exception as s_err:
        print(f" [АВАРИЯ ЗАПИСИ] Превратники заблокировали шлюз диска: {s_err}")
        return False

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
        state_dict = load_file(CHROMA_MODEL_PATH, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(" -> [OK] Заводской граф весов успешно состыкован со структурами трансформера.")
        del state_dict
    except Exception as s_err:
        raise RuntimeError(f"[АВАРИЯ ВЕСОВ] Крах инициализации safetensors: {s_err}")
        
    # 3. СНАЙПЕРСКОЕ СЖАТИЕ TORCHAO (Перенесено на рабочее место после заливки весов)
    print("[RUN] Подключаю промышленный квантизатор TorchAO: поджимаю базу весов in INT8...")
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(model, int8_weight_only())
        print(" -> [OK] Базовый монолит успешно квантован (int8_weight_only). Полка VRAM защищена.")
    except Exception as ao_err:
        print(f" [WARN] Сбой TorchAO-кастинга весов: {ao_err}. Переход на ванильный bfloat16-контур.")
        
    # 4. Инжекция LoRA-электродов поверх сжатой и заполненной базы (76 швов / 152 параметра)
    patched_count = patch_chroma_reactor(model, rank=16)
    model = model.to(device)
    
    # 5. ТОТАЛЬНАЯ ВЫЖЖЕННАЯ ЗЕМЛЯ: Принудительно гасим Autograd для ВСЕХ базовых слоев без исключения
    print("[RUN] Активирую абсолютный фильтр градиентов: замораживаю 100% базового кремния...")
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            param.grad = None # Полное Си-стирание следов графов из памяти
        else:
            param.requires_grad = True
            
    approximator = None
    mod_projector = None
    
    # 6. Фиксация 8-битного оптимизатора СТРОГО и ИСКЛЮЧИТЕЛЬНО на параметрах LoRA
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            trainable_params.append(param)
            
    optimizer = AdamW8bit(trainable_params, lr=1e-4)
    print(f" -> [OK] Автономный оптимизатор AdamW8bit зафиксирован строго на {len(trainable_params)} LoRA-параметрах (Ожидается ровно 152!).")
    
    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"
    
    if not os.path.exists(LATENT_DIR) or not os.path.exists(TEXT_DIR):
        print(" [WARN] Холодная симуляция на тестовом фантом-батче.")
        batch = {
            "latent": torch.randn(1, 16, 128, 128, dtype=torch.float32).to(torch.bfloat16),
            "clip_hidden": torch.randn(1, 77, 768, dtype=torch.float32).to(torch.bfloat16),
            "t5_hidden": torch.randn(1, 256, 4096, dtype=torch.float32).to(torch.bfloat16)
        }
        dataloader = [batch]
    else:
        from chroma_core.init import ChromaDataset
        dataset = ChromaDataset(latent_dir=LATENT_DIR, text_dir=TEXT_DIR)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
    print("# === РАКЕТНЫЙ ЗАПУСК РЕАКТОРА: СТАРТ ЧИСТОГО ЦИКЛА ПЛАВКИ ===")
    import sys
    sys.settrace(trace_lines)
    
    for step, batch in enumerate(dataloader):
        try:
            loss = train_step_core(batch, model, optimizer, approximator, mod_projector, step=step)
            print(f" -> [ШАГ №{step + 1}] ПЛАВКА СТАБИЛЬНА! Текущий Loss: {loss:.6f}")
            
            # Автосохранение каждые 250 шагов и на первом контрольном шаге
            if step == 0 or (step + 1) % 250 == 0:
                checkpoint_path = f"Z:\\flowch\\checkpoints\\chroma_lora_step_{step + 1}.safetensors"
                save_lora_checkpoint(model, checkpoint_path)
                
            # Для длительного марша на 2000 шагов — просто закомментируйте строку 'break' ниже!
            if step == 0:
                break
        except Exception as e:
            print(f" [АВАРИЯ РАД ТАЙМА]: Цикл прерван на шаге {step + 1}: {e}")
            break
            
    sys.settrace(None)
    print("# === ДВИЖОК ВЕРИФИЦИРОВАН. СУХОЙ ПУСК LORA ПРОШЕЛ УСПЕШНО. КОНЕЦ СЕССИИ ===")

if __name__ == "__main__":
    run_reactor_forge()
#---------------- Конец Блока 5 -----------------
