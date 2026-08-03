import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

# Фиксация WDDM-политики Windows против утечек во внешнюю RAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Прямой линк на изолированные узлы очищенного ядра реактора
sys.path.append(os.path.abspath("./src"))
from chroma_core.layers_clean import ChromaTelemetry, Approximator, distribute_modulations
from chroma_core.tensor_math import attention
from ao_optim_monolith import AdamW8bit

# ШОВ-ПЕРЕХВАТЧИК: Вывод каждой исполняемой строки ядра геометрии
def trace_lines(frame, event, arg):
    if event == "line":
        code = frame.f_code
        filename = code.co_filename
        if "chroma_core" in filename or "train_engine" in filename:
            print(f" [TRACE] {filename}:{frame.f_lineno} -> {code.co_name}")
    return trace_lines

class ChromaInlineLoRA(nn.Module):
    """Кастомный шов LoRA. Врезается напрямую в линейные слои."""
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
        ChromaTelemetry.verify(x, f"LoRA_In_{self.rank}")
        base_out = self.base_layer(x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            lora_out = torch.matmul(x.to(torch.bfloat16), self.lora_A)
            lora_out = torch.matmul(lora_out, self.lora_B) * self.scale
            out = base_out + lora_out.to(base_out.dtype)
        ChromaTelemetry.verify(out, f"LoRA_Out_{self.rank}")
        return out

    def verify_gradients(self, layer_name: str):
        for name, param in [("lora_A", self.lora_A), ("lora_B", self.lora_B)]:
            if param.grad is None:
                print(f" [WARN] МЕРТВЫЙ ГРАДИЕНТ [{layer_name}.{name}]: Обновление отсутствует!")
            elif torch.isnan(param.grad).any():
                print(f" [КРАХ] ВЗРЫВ ГРАДИЕНТА [{layer_name}.{name}]: Обнаружен NaN!")

def patch_chroma_reactor(model: nn.Module, rank: int = 16) -> int:
    patched_count = 0
    print("# === ИНИЦИАЛИЗАЦИЯ ИНЖЕКЦИИ АДАПТЕРОВ В ЯДРО RECTOR ===")
    for name, module in model.named_modules():
        if any(target in name for target in ["txt_attn", "img_attn", "linear1"]):
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
                print(f" -> [OK] Инжектирован шов: {name} | База заморожена.")
    if patched_count == 0:
        raise RuntimeError("[АВАРИЯ] Точки инжекции LoRA не найдены!")
    print(f"# === ИНЖЕКЦИЯ ЗАВЕРШЕНА. УСПЕШНО СВАРЕНО ШВОВ: {patched_count} ===")
    return patched_count

def train_step_core(batch: dict, model: nn.Module, optimizer: AdamW8bit, approximator: nn.Module) -> float:
    """Выполняет один боевой шаг плавки LoRA по траекториям Rectified Flow."""
    # ПРИЦЕЛЬНЫЙ ПУСК ДЕФЕКТΟΣΚОПА
    sys.settrace(trace_lines)
    
    x1 = batch["latent"].cuda()
    clip_hidden = batch["clip_hidden"].cuda()
    t5_raw = batch["t5_hidden"].cuda()
    
    # ШОВ ВЫРАВНИВАНИЯ КОНТЕНТА: Добиваем T5-контекст нулями до 512 токенов
    B_pad, L_pad, D_pad = t5_raw.shape
    if L_pad < 512:
        padding_size = 512 - L_pad
        zero_padding = torch.zeros((B_pad, padding_size, D_pad), dtype=t5_raw.dtype, device=t5_raw.device)
        t5_hidden = torch.cat([t5_raw, zero_padding], dim=1)
    else:
        t5_hidden = t5_raw[:, :512, :]
        
    ChromaTelemetry.verify(x1, "train_step.latents", 4)
    optimizer.zero_grad(set_to_none=True)

    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype)
    xt = t.view(-1, 1, 1, 1) * x1 + (1.0 - t.view(-1, 1, 1, 1)) * x0
    target_velocity = x1 - x0

    t_vec = t.view(-1, 1).to(torch.bfloat16).requires_grad_(True)
    monolithic_mod = approximator(t_vec).unsqueeze(1)
    
    flat_mods = distribute_modulations(monolithic_mod, depth_single_blocks=38, depth_double_blocks=19)
    mods = {"double": [], "single": [], "final": None}
    for i in range(19):
        img_mod_pair = flat_mods[f"double_blocks.{i}.img_mod.lin"]
        txt_mod_pair = flat_mods[f"double_blocks.{i}.txt_mod.lin"]
        mods["double"].append([img_mod_pair, txt_mod_pair])
    for i in range(38):
        mods["single"].append(flat_mods[f"single_blocks.{i}.modulation.lin"])
    mods["final"] = flat_mods["final_layer.adaLN_modulation.1"]

    pred_velocity = model(xt, t5_hidden, mods)
    loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
    
    if torch.isnan(loss):
        sys.settrace(None)
        raise ValueError("[КВАНТОВЫЙ ПРОЖОГ] Loss свалился в NaN!")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    for name, module in model.named_modules():
        if hasattr(module, "verify_gradients"):
            module.verify_gradients(name)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    
    # ОТКЛЮЧЕНИЕ ПЕРЕХВАТЧИКА
    sys.settrace(None)
    return loss.item()

def run_reactor_forge():
    print("# === ИНИЦИАЛИЗАЦИЯ ДВИЖКА ТРЕНИРОВКИ TRAIN_ENGINE_V02 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 3072
    num_double = 19
    num_single = 38

    from chroma_core.layers_clean import DoubleStreamBlock, SingleStreamBlock

    class ChromaMMDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_in = nn.Linear(64, hidden_size, dtype=torch.bfloat16)
            self.txt_in = nn.Linear(4096, hidden_size, dtype=torch.bfloat16)
            self.final_layer = nn.Linear(hidden_size, 64, dtype=torch.bfloat16)
            self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden_size) for _ in range(num_double)])
            self.single_blocks = nn.ModuleList([SingleStreamBlock(hidden_size) for _ in range(num_single)])

        def pack_latents(self, x: torch.Tensor) -> torch.Tensor:
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

            for i, block in enumerate(self.double_blocks):
                txt_tokens, img_tokens = block(
                    txt_tokens, img_tokens, None, mods["double"][i]
                )
            
            x_combined = torch.cat([txt_tokens, img_tokens], dim=1)
            for i, block in enumerate(self.single_blocks):
                x_combined = block(
                    x_combined, None, mods["single"][i]
                )

            pred_img_flat = x_combined[:, txt_tokens.shape[1]:]
            pred_img_flat = self.final_layer(pred_img_flat)
            
            B, C_raw, H_raw, W_raw = x_latent.shape
            H, W = H_raw // 2, W_raw // 2
            out = pred_img_flat.view(B, H, W, 16, 2, 2)
            out = out.permute(0, 3, 1, 4, 2, 5)
            return out.reshape(B, 16, H_raw, W_raw)

    model = ChromaMMDiT()
    patched_count = patch_chroma_reactor(model, rank=16)
    model = model.to(device)

    approximator = Approximator(in_dim=1, out_dim=hidden_size, hidden_dim=hidden_size, n_layers=4).to(device)
    optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    print(" -> [OK] 8-битный оптимизатор градиентов AdamW8bit зафиксирован.")

    LATENT_DIR = "./dataset/latent_cache"
    TEXT_DIR = "./dataset/text_cache"

    if not os.path.exists(LATENT_DIR) or not os.path.exists(TEXT_DIR):
        print(" [WARN] Холодная симуляция на фантом-батче.")
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
            loss = train_step_core(batch, model, optimizer, approximator)
            print(f" -> [ШАГ №{step + 1}] ПЛАВКА СТАБИЛЬНА! Текущий Loss: {loss:.6f}")
            if step >= 1:
                break
        except Exception as e:

if __name__ == "__main__":

    run_reactor_forge()
