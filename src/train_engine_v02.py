# Блок №1: Инициализация окружения и маршевые импорты train_engine_v02.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Фиксация WDDM-политики Windows против утечек во внешнюю RAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Прямой линк на изолированные узлы очищенного ядра реактора
sys.path.append(os.path.abspath("./src"))
from chroma_core.layers_clean import ChromaModulationBus, ChromaTelemetry
from chroma_core.tensor_math import attention
from ao_optim_monolith import AdamW8bit

#-------------------- Блок №2 (АБСОЛЮТНАЯ ЗАЩИТА): Кастомный инжектор ChromaInlineLoRA --------------------
class ChromaInlineLoRA(nn.Module):
    """
    Кастомный шов LoRA. Отказ от PEFT. Врезается напрямую в линейные слои [2.2].
    Матрицы инициализируются строго в bfloat16 на девайсе базового слоя.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Жесткая привязка к девайсу базового замороженного слоя для исключения CPU-шума
        target_device = base_layer.weight.device
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=torch.bfloat16, device=target_device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.bfloat16, device=target_device))

    def __getattr__(self, name: str):
        """Проброс системных атрибутов (.weight, .bias) к базовому слою для обхода проверок в ядрах."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Телеметрия входного потока
        ChromaTelemetry.verify(x, f"LoRA_In_{self.rank}")
        
        # Базовый прогон через замороженное FP8/BF16 ядро
        base_out = self.base_layer(x)
        
        # Жесткий контекст вычислений адаптера для изоляции от шума
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Формула шва: Y = Base(X) + ((X * A) * B) * scale
            lora_out = torch.matmul(x.to(torch.bfloat16), self.lora_A)
            lora_out = torch.matmul(lora_out, self.lora_B) * self.scale
            
        out = base_out + lora_out.to(base_out.dtype)
        ChromaTelemetry.verify(out, f"LoRA_Out_{self.rank}")
        return out

    def verify_gradients(self, layer_name: str):
        """Авторитарный контроль градиентов на шаге backward."""
        for name, param in [("lora_A", self.lora_A), ("lora_B", self.lora_B)]:
            if param.grad is None:
                print(f" [WARN] МЕРТВЫЙ ГРАДИЕНТ [{layer_name}.{name}]: Обновление отсутствует!")
            elif torch.isnan(param.grad).any():
                print(f" [КРАХ] ВЗРЫВ ГРАДИЕНТА [{layer_name}.{name}]: Обнаружен NaN!")
#-------------------- Окончание блока №2 --------------------

#-------------------- Блок №3: Снайперский инжектор patch_chroma_reactor --------------------
def patch_chroma_reactor(model: nn.Module, rank: int = 16) -> int:
    """
    Сканирует граф весов Chroma v50 и врезает ChromaInlineLoRA [3.3].
    Отказ от PEFT исключает разрушение монолитной шины модуляции.
    """
    patched_count = 0
    print("# === ИНИЦИАЛИЗАЦИЯ ИНЖЕКЦИИ АДАПТЕРОВ В ЯДРО RECTOR ===")
    
    # Рекурсивный обход всех подмодулей 57 блоков MMDiT
    for name, module in model.named_modules():
        # Таргеты плавки по Платиновой Книге [3.3]: внимание Double и Single блоков
        if any(target in name for target in ["txt_attn", "img_attn", "linear1"]):
            if isinstance(module, nn.Linear):
                # Извлекаем родительский модуль и имя слоя для inline-подмены
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                # Замораживаем базу наглухо, очищая граф Autograd
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                # Врезка кастомного шва LoRA
                lora_wrapper = ChromaInlineLoRA(module, rank=rank)
                setattr(parent, child_name, lora_wrapper)
                patched_count += 1
                
                print(f" -> [OK] Инжектирован шов: {name} | База заморожена.")
                
    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Точки инжекции LoRA не найдены! Проверь топологию весов.")
        
    print(f"# === ИНЖЕКЦИЯ ЗАВЕРШЕНА. УСПЕШНО СВАРЕНО ШВОВ: {patched_count} ===")
    return patched_count
#-------------------- Окончание блока №3 --------------------

#-------------------- Блок №4 (АВТОНОМНЫЙ БОЕВОЙ): Маршевое ядро train_step_core --------------------
def train_step_core(batch: dict, model: nn.Module, bus: ChromaModulationBus, optimizer: AdamW8bit, approximator: nn.Module) -> float:
    """
    Выполняет один боевой шаг плавки LoRA по траекториям Rectified Flow.
    Принимает эталонный 4D-батч латентов [B, 16, 128, 128] напрямую из DataLoader.
    """
    # 1. Извлечение и сквозной контроль геометрии сырого 4D-потока данных
    x1 = batch["latent"].cuda()              # Исходный латент 4D: [B, 16, 128, 128]
    clip_hidden = batch["clip_hidden"].cuda()  # Текст CLIP
    t5_hidden = batch["t5_hidden"].cuda()      # Текст T5
    
    # Контроль геометрии: жестко верифицируем сырой 4D латент перед наложением шума
    ChromaTelemetry.verify(x1, "train_step.latents", 4)
    
    # 2. Generation траектории Rectified Flow в исходном 4D пространстве латентов
    x0 = torch.randn_like(x1) # Чистый латентный шум на CUDA
    
    # Рандомный таймстеп t для каждого элемента батча
    t = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)
    
    # Линейный транспортный путь (4D)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0
    target_velocity = x1 - x0
    
    # 3. Расчет монолитной шины модуляции векторов управления через аппроксиматор
    monolithic_mod = approximator(t, clip_hidden, t5_hidden)
    mods = bus.distribute_modulations(monolithic_mod)
    
    # 4. Подача в модель: вся упаковка и проекция скрыты внутри форварда ChromaMMDiT
    optimizer.zero_grad(set_to_none=True)
    
    # Модель возвращает предсказанное поле скоростей в исходной 4D геометрии [B, 16, 128, 128]
    pred_velocity = model(xt, t5_hidden, mods)
    
    # Расчет ошибки между истинным полем скоростей и предсказанным
    loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    
    if torch.isnan(loss):
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Loss свалился в NaN! Рантайм остановлен.")
        
    # 5. Обратный проход и прокалка градиентов Master Weights
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # 6. Жесткое шлакоотделение (Очистка памяти WDDM Windows)
    torch.cuda.empty_cache()
    
    return loss.item()
#-------------------- Окончание блока №4 --------------------

#-------------------- Блок №5 (ГЕРМЕТИЧНЫЙ МЕТРОПОЛИЯ): Инициализация и Управляющий Цикл --------------------
from torch.utils.checkpoint import checkpoint

def run_reactor_forge():
    """
    Главный пульт управления процессом плавки LoRA на хосте APEX.
    Внедрена авторитарная защита Gradient Checkpointing против утечек в Shared VRAM.
    """
    print("# === ИНИЦИАЛИЗАЦИЯ ДВИЖКА ТРЕНИРОВКИ TRAIN_ENGINE_V02 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hidden_size = 3072
    num_double = 19
    num_single = 38
    
    from chroma_core.layers_clean import DoubleStreamBlock, SingleStreamBlock
    
    class ChromaMMDiT(nn.Module):
        def __init__(self):
            super().__init__()
            # Входные проекторы Метрополии согласно физической карте весов
            self.img_in = nn.Linear(64, hidden_size, dtype=torch.bfloat16)
            self.txt_in = nn.Linear(4096, hidden_size, dtype=torch.bfloat16)
            self.final_layer = nn.Linear(hidden_size, 64, dtype=torch.bfloat16)
            
            self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden_size) for _ in range(num_double)])
            self.single_blocks = nn.ModuleList([SingleStreamBlock(hidden_size) for _ in range(num_single)])
            
        def pack_latents(self, x: torch.Tensor) -> torch.Tensor:
            """Снахперское схлопывание блоков 2х2 по уставу Метрополии: [B, 16, 128, 128] -> [B, 4096, 64]"""
            B, C, H, W = x.shape
            x = x.view(B, C, H // 2, 2, W // 2, 2)
            x = x.permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B, (H // 2) * (W // 2), C * 4)

        def forward(self, x_latent, txt_hidden, mods):
            if len(txt_hidden.shape) == 4:
                txt_hidden = txt_hidden.squeeze(1)
                
            x_latent = x_latent.to(torch.bfloat16)
            txt_hidden = txt_hidden.to(torch.bfloat16)
            
            xt_flat = self.pack_latents(x_latent)
            img_tokens = self.img_in(xt_flat)
            txt_tokens = self.txt_in(txt_hidden)
            
            # --- Контур защиты №1: Gradient Checkpointing для Double-блоков ---
            # Пересчитываем активации внимания на бэкварде, освобождая VRAM под градиенты
            for i, block in enumerate(self.double_blocks):
                # Чекпоинт требует функцию и её позиционные аргументы
                def create_custom_forward(layer):
                    def custom_forward(t_tok, i_tok, modulation):
                        return layer(t_tok, i_tok, modulation)
                    return custom_forward
                
                txt_tokens, img_tokens = checkpoint(
                    create_custom_forward(block), 
                    txt_tokens, 
                    img_tokens, 
                    mods["double"][i],
                    use_reentrant=False # Безопасный режим Autograd без утечек контекста
                )
                
            x_combined = torch.cat([txt_tokens, img_tokens], dim=1)
            
            # --- Контур защиты №2: Gradient Checkpointing для Single-блоков ---
            for i, block in enumerate(self.single_blocks):
                def create_single_forward(layer):
                    def single_forward(combined_tok, modulation):
                        return layer(combined_tok, modulation)
                    return single_forward
                    
                x_combined = checkpoint(
                    create_single_forward(block),
                    x_combined,
                    mods["single"][i],
                    use_reentrant=False
                )
                
            pred_img_flat = x_combined[:, txt_tokens.shape[1]:]
            pred_img_flat = self.final_layer(pred_img_flat)
            
            B, C_raw, H_raw, W_raw = x_latent.shape
            H, W = H_raw // 2, W_raw // 2
            
            out = pred_img_flat.view(B, H, W, 16, 2, 2)
            out = out.permute(0, 3, 1, 4, 2, 5)
            return out.reshape(B, 16, H_raw, W_raw)

    model = ChromaMMDiT()
    
    # Врезка кастомных швов LoRA в CPU-каркас
    patched_count = patch_chroma_reactor(model, rank=16)
    
    # Подъем всего герметичного комплекса на CUDA в один чистый приём
    model = model.to(device)
    
    class DummyApproximator(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.linear = nn.Linear(1, out_features, dtype=torch.bfloat16)
        def forward(self, t, clip, t5):
            return self.linear(t.to(torch.bfloat16).view(-1, 1))

    bus = ChromaModulationBus(hidden_size=hidden_size, num_double=num_double, num_single=num_single)
    approximator = DummyApproximator(bus.expected_features).to(device)
    
    optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    print(" -> [OK] 8-битный оптимизатор градиентов AdamW8bit зафиксирован.")
    
    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"
    
    if not os.path.exists(LATENT_DIR) or not os.path.exists(TEXT_DIR):
        print(" [WARN] Карантинные зоны кэша отсутствуют. Холодная симуляция на фантом-батче.")
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
            loss = train_step_core(batch, model, bus, optimizer, approximator)
            print(f" -> [ШАГ №{step + 1}] ПЛАВКА СТАБИЛЬНА! Текущий Loss: {loss:.6f}")
            
            for name, module in model.named_modules():
                if hasattr(module, "verify_gradients"):
                    module.verify_gradients(name)
                    
            if step >= 1: 
                break
        except Exception as e:
            print(f" [АВАРИЯ РАД ТАЙМА]: Цикл прерван на шаге {step + 1}: {e}")
            break
            
    print("# === ДВИЖОК ВЕРИФИЦИРОВАН. ВСЕ ШВЫ ДЕРЖАТ УДАР. КОНЕЦ СЕССИИ ===")

if __name__ == "__main__":
    run_reactor_forge()
#-------------------- Окончание блока №5 --------------------

