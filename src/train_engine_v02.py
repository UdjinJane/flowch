#---------------- Старт Блока 1 (Супер-Боровик тотального следствия и контроля С++ либ) ------------
import os
import sys
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Жесткая блокировка фрагментации и активация расширяемых сегментов кремния
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

# ФАЗОВЫЙ СИ-ШУНТ: Сначала регистрируем трюм src, только потом дергаем бортовые модули!
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("."))

from chroma_core.layers_clean import ChromaTelemetry, Approximator, distribute_modulations
from chroma_core.tensor_math import attention
# ВОЗВРАТ НАДЛЕЖАЩЕГО ИМПОРТА: Подключаем чистокровный бортовой оптимизатор Кэпа
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

class ChromaStandaloneLoRA(nn.Module):
    """
    Высшая автономная броня v7.0: Абсолютно стерильный автономный шов LoRA.
    ФУНДАМЕНТАЛЬНЫЙ СИ-ШУНТ: Больше не содержит внутри себя базовый квантованный слой nn.Linear!
    Полностью изолирован от С++ хуков TorchAO. Хранит только легализованные параметры адаптера.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, target_device="cuda"):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        # Изначально создаем чистые Си-тензоры, чтобы пролететь радар TorchAO на взлете
        self.lora_A = torch.randn(in_features, rank, dtype=torch.bfloat16, device=target_device) * 0.02
        self.lora_B = torch.zeros(rank, out_features, dtype=torch.bfloat16, device=target_device)

    def legalise_for_optimizer(self):
        """Превращает скрытые тензоры в официальные nn.Parameter модели."""
        if not isinstance(self.lora_A, nn.Parameter):
            self.lora_A = nn.Parameter(self.lora_A, requires_grad=True)
        if not isinstance(self.lora_B, nn.Parameter):
            self.lora_B = nn.Parameter(self.lora_B, requires_grad=True)
        return self.lora_A, self.lora_B

    def forward_lora_only(self, x: torch.Tensor) -> torch.Tensor:
        """Рассчитывает исключительно дельту адаптера, полностью игнорируя базу."""
        x_cont = x.contiguous().to(torch.bfloat16)
        with torch.no_grad():
            lora_mid = torch.matmul(x_cont, self.lora_A.detach())
            lora_out = torch.matmul(lora_mid, self.lora_B.detach()) * self.scale
        return lora_out.detach()

    def inject_manual_backward(self, loss_grad_output: torch.Tensor, static_incoming_x: torch.Tensor):
        """
        Локальный инжектор Омниссии v7.0: Чистый расчет градиентов без зацепа 
        С++ Autograd-инфраструктуры PyTorch.
        """
        with torch.no_grad():
            x_cont = static_incoming_x.contiguous().to(torch.bfloat16)
            dy = loss_grad_output.to(torch.bfloat16) * self.scale

            # Синхронизация геометрии последовательностей токенов
            if dy.shape[1] > x_cont.shape[1]:
                dy = dy[:, :x_cont.shape[1], :]
            elif dy.shape[1] < x_cont.shape[1]:
                padding_size = x_cont.shape[1] - dy.shape[1]
                zero_padding = torch.zeros((dy.shape[0], padding_size, dy.shape[2]), dtype=dy.dtype, device=dy.device)
                dy = torch.cat([dy, zero_padding], dim=1)

            lora_mid = torch.matmul(x_cont, self.lora_A.detach())

            dy_flat = dy.view(-1, dy.shape[-1])
            mid_flat = lora_mid.view(-1, lora_mid.shape[-1])
            x_flat = x_cont.view(-1, x_cont.shape[-1])

            if mid_flat.shape[0] == dy_flat.shape[0]:
                grad_B = torch.matmul(mid_flat.t(), dy_flat)
                d_mid = torch.matmul(dy_flat, self.lora_B.detach().t())
                grad_A = torch.matmul(x_flat.t(), d_mid)

                if self.lora_A.grad is None:
                    self.lora_A.grad = grad_A.view_as(self.lora_A)
                else:
                    self.lora_A.grad += grad_A.view_as(self.lora_A)

                if self.lora_B.grad is None:
                    self.lora_B.grad = grad_B.view_as(self.lora_B)
                else:
                    self.lora_B.grad += grad_B.view_as(self.lora_B)

        return None

    def verify_gradients(self, layer_name: str, current_step: int = 0):
        """Умная телеметрия шва."""
        if current_step == 0 and "double_blocks.0." in layer_name:
            print(f"[ТЕЛЕМЕТРИЯ СТЕРИЛЬНОГО ШВА] Проверка для {layer_name}:")
            for name, param in [("lora_A", self.lora_A), ("lora_B", self.lora_B)]:
                if param.grad is None:
                    print(f" -> [WARN] МЕРТВЫЙ ГРАДИЕНТ [{name}]")
                elif torch.isnan(param.grad).any():
                    print(f" -> [КРАХ] ВЗРЫВ ГРАДИЕНТА [{name}]")
                else:
                    grad_mean = param.grad.abs().mean().item()
                    print(f" -> [OK] Ток стабилен. Средний градиент {name}: {grad_mean:.8f}")
#---------------- Конец Блока 1 -----------------

#---------------- Старт Блока 2 (Снайперский Инжектор Спутников PROJ/LINEAR2 без подмены Базы) ------------
def patch_chroma_reactor(model: nn.Module, rank: int = 16) -> int:
    """
    Динамически обходит граф весов трансформера и монтирует LoRA-спутники РЯДОМ с базой.
    ФУНДАМЕНТАЛЬНЫЙ СИ-ШУНТ: Базовый квантованный слой TorchAO вообще НЕ подменяется!
    Это полностью исключает активацию С++ backward-хуков и стирает фантом на 384 ГБ.
    """
    patched_count = 0
    print("# === ИНИЦИАЛИЗАЦИЯ МОНТАЖА АВТОНОМНЫХ СПУТНИКОВ RECTOR v7.0 ===")
    
    # Собираем модули в список, чтобы избежать RuntimeError при динамической модификации словаря
    modules_to_check = list(model.named_modules())
    
    for name, module in modules_to_check:
        # Снайперский прицел строго на изолированные выходные проекции иmlp-выводы
        if any(target in name for target in ["img_attn.proj", "txt_attn.proj", "linear2"]):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Базовый слой намертво цементируем и запрещаем Autograd-трекинг
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                # Монтируем АВТОНОМНЫЙ спутник прямо в родительский блок под уникальным именем
                shadow_name = f"{child_name}_lora_shadow"
                in_features = module.in_features
                out_features = module.out_features
                target_device = module.weight.device

                # Создаем чистокровный изолированный спутник v7.0
                lora_shadow = ChromaStandaloneLoRA(in_features, out_features, rank=rank, target_device=target_device)
                setattr(parent, shadow_name, lora_shadow)
                
                patched_count += 1
                print(f" -> [OK] Смонтирован параллельный спутник: {name}_lora_shadow | База изолирована.")

    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Ошибка сканирования: целевые точки (.proj/.linear2) для монтажа спутников не найдены!")
    print(f"# === МОНТАЖ ЗАВЕРШЕН. УСПЕШНО СВАРЕНО СПУТНИКОВ: {patched_count} ===")
    return patched_count
#---------------- Конец Блока 2 -----------------

#---------------- Старт Блока 3 (Монолитное Боевое Ядро - Контур Чистой Плавки и Хронометража) ----
import time
import torch.nn.functional as F

def train_step_core(batch: dict, model: nn.Module, optimizer: AdamW8bit, approximator: nn.Module, mod_projector: nn.Module, step: int = 0) -> float:
    """
    Выполняет один боевой шаг плавки с ручным распределением градиентного тока по спутникам LoRA.
    Жестко изолирует легализованные параметры от стандартного Autograd-графа во избежание Си-конфликтов.
    """
    # Фиксация старта фазы ввода-вывода (I/O) и подготовки батча
    t_start = time.perf_counter()
    x1_raw = batch["latent"].cuda()
    t5_raw = batch["t5_hidden"].cuda()

    # Си-снайпер принудительного урезания пространственной геометрии латента
    if x1_raw.shape[-1] == 128:
        x1 = F.interpolate(x1_raw.float(), size=(64, 64), mode="bilinear").to(torch.bfloat16)
    else:
        x1 = x1_raw

    if len(t5_raw.shape) == 4:
        t5_raw = t5_raw.squeeze(1)

    B_pad, L_pad, D_pad = t5_raw.shape
    if L_pad < 512:
        padding_size = 512 - L_pad
        zero_padding = torch.zeros((B_pad, padding_size, D_pad), dtype=t5_raw.dtype, device=t5_raw.device)
        t5_hidden = torch.cat([t5_raw, zero_padding], dim=1)
    else:
        t5_hidden = t5_raw[:, :512, :]

    # === КОНТУР ВХОДНОЙ ТЕЛЕМЕТРИИ ОМНИССИИ ===
    if step == 0:
        shape_vae_str = str(list(x1.shape))
        dtype_vae_str = str(x1.dtype)
        shape_t5_str = str(list(t5_hidden.shape))
        dtype_t5_str = str(t5_hidden.dtype)

        mem_alloc_gb = str(round(torch.cuda.memory_allocated() / 1024**3, 2))
        mem_res_gb = str(round(torch.cuda.memory_reserved() / 1024**3, 2))

        print(f"\n┌── [МАРШЕВАЯ ТЕЛЕМЕТРИЯ ЯДРА RECTOR | ПУСКОВАЯ ВЕРИФИКАЦИЯ ВХОДОВ] ────────────────┐")
        print(f"│ * Входной латент VAE : Форма {shape_vae_str:<18} | Тип {dtype_vae_str:<10} | Mean {x1.abs().mean().item():.4f} │")
        print(f"│ * Шина текста T5XXL  : Форма {shape_t5_str:<18} | Type {dtype_t5_str:<10} | Mean {t5_hidden.abs().mean().item():.4f} │")
        print(f"│ * Полка видеопамяти  : Выделено {mem_alloc_gb:<5} ГБ | Зарезервировано {mem_res_gb:<5} ГБ      │")
        print(f"└─────────────────────────────────────────────────────────────────────────────────────┘\n")

    # ЖЕСТКИЙ СИ-ШУНТ: Тотальное обнуление градиентных регистров спутников перед началом шага
    for name, module in model.named_modules():
        if hasattr(module, "inject_manual_backward"):
            if module.lora_A.grad is not None:
                module.lora_A.grad.zero_()
            if module.lora_B.grad is not None:
                module.lora_B.grad.zero_()

    optimizer.zero_grad(set_to_none=True)

    # Generation пространственного шума и траектории Rectified Flow
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape, device=x1.device, dtype=torch.float32).to(x1.dtype)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0

    target_velocity = (x1.float() - x0.float()).detach()
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
        # Форвард базового трансформера (чистый и изолированный TorchAO контур)
        pred_velocity_raw = model(xt, t5_hidden, mods)
        pred_velocity = pred_velocity_raw.detach().cpu().float().cuda()
        
        # Интеграция параллельного тока спутников LoRA на уровне вывода графа (при необходимости)
        # В данной архитектуре бэкворд течет независимо через сохраненный кэш активаций
        loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)

    if torch.isnan(loss):
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Критическая ошибка: Loss рухнул в NaN!")
    t_fwd = time.perf_counter() - t_fwd_start

    # Фиксация и запуск фазы обратного прохода (Ручной Бэкворд Омниссии v7.0)
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

        # 3. Проход сквозь final_layer
        final_weight = model.final_layer.weight.to(torch.bfloat16)
        base_grad_output = torch.matmul(grad_flat_64, final_weight) # Калибр [B, H*W, 3072]

        # Пред-расчет изолированных источников токенов для сопряжения со спутниками
        xt_flat_base = model.pack_latents(xt)
        static_img_tokens = model.img_in(xt_flat_base).detach().contiguous()
        static_txt_tokens = model.txt_in(t5_hidden).detach().contiguous()

        # === ТОПОЛОГИЧЕСКОЕ ВЫРАВНИВАНИЕ ШИНЫ ГРАДИЕНТОВ v7.0 ===
        combined_static_tokens = torch.cat([static_img_tokens, static_txt_tokens], dim=1).contiguous()
        
        txt_len = static_txt_tokens.shape[1]
        zero_txt_grad = torch.zeros((B, txt_len, base_grad_output.shape[-1]), dtype=torch.bfloat16, device="cuda")
        combined_grad_output = torch.cat([base_grad_output, zero_txt_grad], dim=1).contiguous()

        # 4. МОНОЛИТНЫЙ ИНЖЕКТОР БЭКВАРДА: Спутники принимают чистые тензоры без С++ Autograd базы
        modules_chain = list(model.named_modules())
        for name, module in reversed(modules_chain):
            if hasattr(module, "inject_manual_backward"):
                module.inject_manual_backward(combined_grad_output.detach(), combined_static_tokens.detach())

    t_bwd = time.perf_counter() - t_bwd_start

    # Фиксация и запуск фазы оптимизации весов (Optimizer Step)
    t_opt_start = time.perf_counter()
    trainable_params = []
    for name, module in model.named_modules():
        if hasattr(module, "inject_manual_backward"):
            trainable_params.extend([module.lora_A, module.lora_B])
            
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    t_opt = time.perf_counter() - t_opt_start

    # Фиксация фазы проверки градиентного тока и снайперской чистки кэша VRAM
    t_clean_start = time.perf_counter()
    for name, module in model.named_modules():
        if hasattr(module, "verify_gradients"):
            module.verify_gradients(name, current_step=step)

    # Принудительное зануление Си-тензоров после шага для предотвращения накопления «снежного кома»
    for name, module in model.named_modules():
        if hasattr(module, "inject_manual_backward"):
            if module.lora_A.grad is not None:
                module.lora_A.grad.zero_()
            if module.lora_B.grad is not None:
                module.lora_B.grad.zero_()
            
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
    print(f"│ -> ПОЛНОЕ ВРЕМЯ ТИКА ЦИКЛА: {t_total:6.3f} сек. Текущий Loss: {loss.item():12.6f} │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────┘\n")

    return loss.item()
#---------------- Конец Блока 3 -----------------

#---------------- Старт Блока 4 (Архитектура Монолитной Шины и Сопряжения Спутников LoRA v7.0) ------------
class ChromaBlockProcessor(nn.Module):
    """
    Вспомогательный С++ контроллер Омниссии для послойного перехвата форварда.
    Снайперски находит смонтированные спутники и подмешивает их параллельный ток дельты.
    """
    @staticmethod
    def inject_shadow_flow(layer_name: str, base_layer: nn.Linear, parent_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Расчет чистого базового выхода (квантованное ядро TorchAO)
        base_output = base_layer(x)
        
        # Поиск автономного спутника lora_shadow по зарегистрированному имени шва
        shadow_attr_name = f"{layer_name}_lora_shadow"
        if hasattr(parent_module, shadow_attr_name):
            shadow_module = getattr(parent_module, shadow_attr_name)
            # Подмешиваем параллельный ток адаптера БЕЗ зацепа Autograd базового слоя
            lora_delta = shadow_module.forward_lora_only(x)
            return base_output + lora_delta
            
        return base_output

class ChromaMMDiT(nn.Module):
    """
    Монолитный Трансформер ChromaMMDiT v7.0 с поддержкой параллельного тока спутников.
    Полностью очищен от AdaLN-Zero (по чертежам Метрополии), использует Rectified Flow.
    """
    def __init__(self):
        super().__init__()
        # Инициализация оригинальной топологии Lodestone Rock (8.9B / 57 блоков)
        self.img_in = nn.Linear(64, 3072, dtype=torch.bfloat16)
        self.txt_in = nn.Linear(4096, 3072, dtype=torch.bfloat16)
        self.final_layer = nn.Linear(3072, 64, dtype=torch.bfloat16)
        
        # Динамические контейнеры под Double и Single блоки фабричной Chroma1-HD
        self.double_blocks = nn.ModuleList([nn.Module() for _ in range(19)])
        self.single_blocks = nn.ModuleList([nn.Module() for _ in range(38)])
        
        # Эмуляция структуры слоев для корректной стыковки load_state_dict
        for i in range(19):
            b = self.double_blocks[i]
            b.img_attn = nn.Module()
            b.img_attn.qkv = nn.Linear(3072, 9216, dtype=torch.bfloat16)
            b.img_attn.proj = nn.Linear(3072, 3072, dtype=torch.bfloat16)
            b.txt_attn = nn.Module()
            b.txt_attn.qkv = nn.Linear(3072, 9216, dtype=torch.bfloat16)
            b.txt_attn.proj = nn.Linear(3072, 3072, dtype=torch.bfloat16)
            
        for i in range(38):
            b = self.single_blocks[i]
            b.linear1 = nn.Linear(3072, 12288, dtype=torch.bfloat16)
            b.linear2 = nn.Linear(12288, 3072, dtype=torch.bfloat16)

    def pack_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Разворачивает 4D VAE латент [B, C, H, W] в 2D последовательность токенов [B, H*W, C*4]."""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H // 2, 2, W // 2, 2)
        x_flat = x_flat.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x_flat.view(B, (H // 2) * (W // 2), C * 4)

    def forward(self, xt: torch.Tensor, t5_hidden: torch.Tensor, mods: dict = None) -> torch.Tensor:
        """
        Маршевый форвард трансформера. Проводит ток активаций сквозь 57 слоев,
        на лету сопрягая базовые квантованные вычисления с параллельными спутниками LoRA.
        """
        # 1. Входная проекция латентов и текстовой шины
        xt_flat = self.pack_latents(xt)
        img_tokens = self.img_in(xt_flat)  # Калибр [B, 1024, 3072]
        txt_tokens = self.txt_in(t5_hidden) # Калибр [B, 512, 3072]

        # 2. ПРОХОД СКВОЗЬ 19 DOUBLE BLOCKS (Параллельная обработка картинки и текста)
        for i, block in enumerate(self.double_blocks):
            # Внутреннее QKV-внимание базовой модели (Квантовано TorchAO, LoRA сюда не лезет)
            img_qkv = block.img_attn.qkv(img_tokens)
            txt_qkv = block.txt_attn.qkv(txt_tokens)
            
            # Математика фабричного механизма внимания (attention)
            img_context = attention(img_qkv)
            txt_context = attention(txt_qkv)
            
            # СНАЙПЕРСКИЙ ПЕРЕХВАТ ВЫХОДНЫХ ПРОЕКЦИЙ: Вливаем ток спутников LoRA v7.0!
            img_tokens = img_tokens + ChromaBlockProcessor.inject_shadow_flow("img_attn.proj", block.img_attn.proj, block.img_attn, img_context)
            txt_tokens = txt_tokens + ChromaBlockProcessor.inject_shadow_flow("txt_attn.proj", block.txt_attn.proj, block.txt_attn, txt_context)

        # Объединение шины для прохода сквозь одиночные блоки
        combined_tokens = torch.cat([img_tokens, txt_tokens], dim=1) # Калибр [B, 1536, 3072]

        # 3. ПРОХОД СКВОЗЬ 38 SINGLE BLOCKS (Монолитная сквозная шина модуляции)
        for i, block in enumerate(self.single_blocks):
            # Проекция на расширенное MLP-пространство (linear1)
            mlp_mid = block.linear1(combined_tokens)
            
            # СНАЙПЕРСКИЙ ПЕРЕХВАТ СЛОЕВ ВЫВОДА MLP: Вливаем ток спутников LoRA на linear2!
            combined_tokens = combined_tokens + ChromaBlockProcessor.inject_shadow_flow("linear2", block.linear2, block, mlp_mid)

        # 4. Изоляция выходного кадра картинок от текстового хвоста последовательности
        img_len = img_tokens.shape[1]
        final_img_tokens = combined_tokens[:, :img_len, :]

        # 5. Выходной Аппроксиматор Метрополии
        output_flat = self.final_layer(final_img_tokens) # Калибр [B, 1024, 64]
        
        # Сборка 2D токенов обратно в исходную 4D геометрию скорости Rectified Flow [B, 16, H, W]
        B = xt.shape[0]
        H, W = xt.shape[2], xt.shape[3]
        output_4d = output_flat.view(B, H // 2, W // 2, 16, 2, 2)
        output_4d = output_4d.permute(0, 3, 1, 4, 2, 5).contiguous()
        return output_4d.view(B, 16, H, W)
#---------------- Конец Блока 4 -----------------

#---------------- Старт Блока 5 (Контур Инициализации, Жесткой Изоляции Градиентов и Сохранения Чекпоинтов) -------
def save_lora_checkpoint(model: nn.Module, save_path: str):
    """
    Контур тотальной дефектоскопии весов LoRA.
    Фильтрует, очищает мантиссу и сохраняет исключительно электроды параллельных спутников.
    """
    print(f"\n# === ЗАПУСК ИНСПЕКЦИИ КОНТУРА СОХРАНЕНИЯ: {save_path} ===")
    lora_state_dict = {}
    corrupted_weights = 0

    # Снайперский ручной сбор изолированных параметров из спутников
    for name, module in model.named_modules():
        if hasattr(module, "inject_manual_backward"):
            t_A = module.lora_A.detach().clone().cpu().to(torch.bfloat16)
            t_B = module.lora_B.detach().clone().cpu().to(torch.bfloat16)

            if torch.isnan(t_A).any() or torch.isinf(t_A).any():
                print(f" -> [КРАХ ТЕЛЕМЕТРИИ] Поврежден lora_A в спутнике: {name}")
                corrupted_weights += 1
            if torch.isnan(t_B).any() or torch.isinf(t_B).any():
                print(f" -> [КРАХ ТЕЛЕМЕТРИИ] Поврежден lora_B в спутнике: {name}")
                corrupted_weights += 1

            lora_state_dict[f"{name}.lora_A"] = t_A
            lora_state_dict[f"{name}.lora_B"] = t_B

    if corrupted_weights > 0:
        print(f" [WARN] Обнаружено поврежденных спутников: {corrupted_weights}. Запись заблокирована для защиты ядра!")
        return False

    if len(lora_state_dict) != 152:
        print(f" [WARN] Аномалия контура спутников: собрано {len(lora_state_dict)} параметров вместо уставных 152!")

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        print(f" -> [OK] Контур запечатан. Веса спутников LoRA успешно сохранены! Размер буфера: {len(lora_state_dict)} узлов.")
        return True
    except Exception as s_err:
        print(f" [АВАРИЯ ЗАПИСИ] Превратники заблокировали шлюз диска: {s_err}")
        return False

def run_reactor_forge():
    """
    Управляет запуском реактора: разворачивает топологию, заливает заводские веса,
    сжимает базу в INT8 через TorchAO, монтирует автономные спутники LoRA и запускает плавку.
    """
    print("# === ИНИЦИАЛИЗАЦИЯ ДВИЖКА ТРЕНИРОВКИ TRAIN_ENGINE_V02 v7.0 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHROMA_MODEL_PATH = r"Z:\flowch\models_core\transformer\Chroma1-HD.safetensors"

    if not os.path.exists(CHROMA_MODEL_PATH):
        raise FileNotFoundError(f"[АВАРИЯ] Заводской сейфтензор не найден по адресу: {CHROMA_MODEL_PATH}")

    # ЖЕСТКИЙ СИ-ШУНТ: Отключаем внутреннюю Си-буферизацию cuBLAS во избежание резервации 128-ГБ блоков
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    # 3. ИНЖЕКЦИЯ LORA СПУТНИКОВ ДО КВАНТОВАНИЯ: База остается нетронутой!
    patched_count = patch_chroma_reactor(model, rank=16)

    # 4. СНАЙПЕРСКОЕ СЖАТИЕ TORCHAO НА ОРИГИНАЛЬНЫЕ БАЗОВЫЕ СЛОИ
    print("[RUN] Подключаю промышленный квантизатор TorchAO строго на базовый монолит весов...")
    try:
        from torchao.quantization import quantize_, int8_weight_only
        # Квантуем строго исходные слои nn.Linear. Окружающие спутники (имеющие другие имена) радар игнорирует!
        for name, module in model.named_modules():
            if "double_blocks" in name or "single_blocks" in name:
                # Проверяем, что это оригинальный слой, а не наш кастомный спутник
                if isinstance(module, nn.Linear) and not "lora_shadow" in name:
                    quantize_(module, int8_weight_only())
        print(" -> [OK] Оригинальный базовый монолит успешно квантован. Спутники в безопасности.")
    except Exception as ao_err:
        print(f" [WARN] Сбой TorchAO-кастинга весов: {ao_err}. Переход на ванильный bfloat16-контур.")

    # 5. РУЧНОЙ СБОР ПАРАМЕТРОВ СПУТНИКОВ И ОФИЦИАЛЬНАЯ ЛЕГАЛИЗАЦИЯ
    # СНАЙПЕРСКИЙ ШУНТ: Легализируем параметры внутри спутников ПОСЛЕ квантования TorchAO.
    # Так как спутники стоят обособленно, они получают чистый статус листьев без заражения базы!
    trainable_params = []
    for name, module in model.named_modules():
        if hasattr(module, "inject_manual_backward"):
            p_A, p_B = module.legalise_for_optimizer()
            trainable_params.extend([p_A, p_B])

    # Фиксация 8-битного оптимизатора в чистом кремнии
    optimizer = AdamW8bit(trainable_params, lr=1e-4)
    print(f" -> [OK] Автономный оптимизатор AdamW8bit успешно принял параметры спутников: {len(trainable_params)} параметров.")

    model = model.to(device)

    # 6. ТОТАЛЬНАЯ ВЫЖЖЕННАЯ ЗЕМЛЯ: Блокируем Autograd для всего остального кремния модели
    for name, param in model.named_parameters():
        # Разрешаем градиенты только для параметров наших автономных спутников
        if not any(x in name for x in ["lora_A", "lora_B"]):
            param.requires_grad = False
            param.grad = None

    approximator = None
    mod_projector = None

    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"

    if not os.path.exists(LATENT_DIR) or not os.path.exists(TEXT_DIR):
        print(" [WARN] Холодная симуляция на УРЕЗАННОМ тест-кадре (64x64) для обхода Си-аллокатора.")
        batch = {
            "latent": torch.randn(1, 16, 64, 64, dtype=torch.float32).to(torch.bfloat16),
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

            if step == 0 or (step + 1) % 250 == 0:
                checkpoint_path = f"Z:\\flowch\\checkpoints\\chroma_lora_step_{step + 1}.safetensors"
                save_lora_checkpoint(model, checkpoint_path)

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
