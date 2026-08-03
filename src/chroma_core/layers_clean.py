#-------------------- Блок №1: Хедер, Телеметрия и Модуляционные Контейнеры --------------------
"""
================================================================================
МАРШЕВОЕ ОЧИЩЕННОЕ ЯДРО ГЕОМЕТРИИ РЕАКТОРА CHROMA V50
================================================================================
ФУНКЦИОНАЛЬНОСТЬ:
- Обеспечивает сквозной рантайм-контроль тензоров (Телеметрия Краха/Прожога).
- Реализует оригинальные типы данных Метрополии (Контейнеры модуляции ModulationOut).
- Выполняет послойную нормализацию RMSNorm с защитой от Underflow.
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class ModulationOut:
    """Оригинальный контейнер векторов управления Метрополии."""
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor

class ChromaTelemetry:
    @staticmethod
    def verify(tensor: torch.Tensor, name: str, expected_dim: int = None):
        """Авторитарный инспектор геометрии и чистоты активаций в CUDA-контуре."""
        if not tensor.is_cuda:
            print(f" ТЕЛЕМЕТРИЯ КРАХА [{name}]: Тензор вылетел из CUDA в системную RAM!")
        if expected_dim and len(tensor.shape) != expected_dim:
            raise ValueError(f" АНОМАЛИЯ РАЗМЕРНОСТИ [{name}]: Ожидалось {expected_dim}D, зафиксировано {tensor.shape}")
        if torch.isnan(tensor).any():
            raise ValueError(f" КВАНТОВЫЙ ПРОЖОГ [{name}]: Зафиксирован деструктивный NaN!")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, "RMSNorm.input")
        x_dtype = x.dtype
        # Апкаст во float32 для исключения численного взрыва при расчете среднего квадрата
        x_f32 = x.float()
        rrms = torch.rsqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        out = (x_f32 * rrms).to(dtype=x_dtype) * self.scale.to(x_dtype)
        ChromaTelemetry.verify(out, "RMSNorm.output")
        return out
#-------------------- Конец блока №1 ----------------------------
#-------------------- Блок №2: Маршевые Эмбеддеры и Изолированный DCT NeRF --------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Генерация синусоидальных эмбеддингов временных координат (Rectified Flow)."""
    ChromaTelemetry.verify(timesteps, "timestep_embedding.input")
    t = time_factor * timesteps
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t.dtype)
    return embedding

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, "MLPEmbedder.input")
        return self.out_layer(self.silu(self.in_layer(x)))

class NerfEmbedder(nn.Module):
    """Оригинальный пространственный DCT-эмбеддер Метрополии с LRU-запечатыванием сеток."""
    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )
        self._cache = {}

    def _fetch_pos(self, patch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = f"dct_{patch_size}_{device}_{dtype}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        
        freqs = torch.linspace(0, self.max_freqs - 1, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        if len(self._cache) >= 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = dct
        return dct

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, P2, C = inputs.shape
        original_dtype = inputs.dtype
        # Жесткая float32 изоляция расчетного поля для исключения PCIe-оверлоада
        with torch.amp.autocast("cuda", enabled=False):
            patch_size = int(P2 ** 0.5)
            inputs_f32 = inputs.float()
            dct = self._fetch_pos(patch_size, inputs.device, torch.float32)
            dct = dct.repeat(B, 1, 1)
            combined = torch.cat([inputs_f32, dct], dim=-1)
            out = self.embedder.float()(combined)
        return out.to(original_dtype)
#-------------------- Конец блока №2 ----------------------------
#---------------- Старт Блока 3 (Оригинальный Аппроксиматор и Пассивный Распределитель Шины Модуляции)---
class Approximator(nn.Module):
    """Многослойный Аппроксиматор Метрополии для генерации векторов управления."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, "Approximator.input")
        x = x.to(self.in_proj.weight.dtype)
        x = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        out = self.out_proj(x)
        ChromaTelemetry.verify(out, "Approximator.output")
        return out

def distribute_modulations(tensor: torch.Tensor, depth_single_blocks: int, depth_double_blocks: int) -> dict:
    """
    Снайперски нарезает монолитный тензор Аппроксиматора на строго
    именованные контейнеры ModulationOut для всех 57 слоев.
    Вычищен от С++ дедлоков chunk(). Нарезка идет по жестким физическим срезам.
    """
    ChromaTelemetry.verify(tensor, "distribute_modulations.input", 3)
    batch_size, vectors, dim = tensor.shape
    block_map = {}
    
    idx = 0
    
    # 1. Нарезка для одиночных блоков (SingleStreamBlocks): 38 блоков по 1 контейнеру
    for i in range(depth_single_blocks):
        block_map[f"single_blocks.{i}.modulation.lin"] = ModulationOut(
            shift=tensor[:, idx : idx + 1, :],
            scale=tensor[:, idx + 1 : idx + 2, :],
            gate=tensor[:, idx + 2 : idx + 3, :]
        )
        idx += 3

    # 2. Нарезка для спаренных блоков (DoubleStreamBlocks): 19 блоков по 2 контейнера (img и txt)
    for i in range(depth_double_blocks):
        # Графический поток управления (2 набора векторов AdaLN-Zero)
        img_mods = [
            ModulationOut(shift=tensor[:, idx : idx + 1, :], scale=tensor[:, idx + 1 : idx + 2, :], gate=tensor[:, idx + 2 : idx + 3, :]),
            ModulationOut(shift=tensor[:, idx + 3 : idx + 4, :], scale=tensor[:, idx + 4 : idx + 5, :], gate=tensor[:, idx + 5 : idx + 6, :])
        ]
        idx += 6
        block_map[f"double_blocks.{i}.img_mod.lin"] = img_mods

        # Текстовый поток управления (2 набора векторов AdaLN-Zero)
        txt_mods = [
            ModulationOut(shift=tensor[:, idx : idx + 1, :], scale=tensor[:, idx + 1 : idx + 2, :], gate=tensor[:, idx + 2 : idx + 3, :]),
            ModulationOut(shift=tensor[:, idx + 3 : idx + 4, :], scale=tensor[:, idx + 4 : idx + 5, :], gate=tensor[:, idx + 5 : idx + 6, :])
        ]
        idx += 6
        block_map[f"double_blocks.{i}.txt_mod.lin"] = txt_mods

    # 3. Финальный слой нормализации AdaLN
    block_map["final_layer.adaLN_modulation.1"] = [
        tensor[:, idx : idx + 1, :],
        tensor[:, idx + 1 : idx + 2, :]
    ]
    
    return block_map
#---------------- Конец Блока 3 -----------------

#---------------- Маршевый Блок №4 (МАРШЕВЫЙ ГЕОМЕТРИЧЕСКИЙ): DoubleStreamBlock и SingleStreamBlock ----------------
class SelfAttention(nn.Module):
    """Изолированный узел внимания блоков Метрополии."""
    def __init__(self, dim: int, num_heads: int = 24, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=torch.bfloat16)
        self.proj = nn.Linear(dim, dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DoubleStreamBlock(nn.Module):
    """Маршевый спаренный блок обработки текстового и графического потоков."""
    def __init__(self, hidden_size: int, num_heads: int = 24, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=torch.bfloat16),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=torch.bfloat16),
        )
        
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=torch.bfloat16),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=torch.bfloat16),
        )

    def forward(self, txt: torch.Tensor, img: torch.Tensor, pe: torch.Tensor, distill_vec: list, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        ChromaTelemetry.verify(img, "DoubleStreamBlock.img_in")
        target_dtype = self.txt_attn.qkv.weight.dtype
        img = img.to(target_dtype)
        txt = txt.to(target_dtype)
        
        img_mod1, img_mod2 = distill_vec[0]
        txt_mod1, txt_mod2 = distill_vec[1]
        
        # Модуляция AdaLN-Zero со сжатием размерности по оси векторов (.squeeze(1) убирает фальшивый шаг)
        img_modulated = (1 + img_mod1.scale.squeeze(1).to(target_dtype)) * self.img_norm1(img) + img_mod1.shift.squeeze(1).to(target_dtype)
        txt_modulated = (1 + txt_mod1.scale.squeeze(1).to(target_dtype)) * self.txt_norm1(txt) + txt_mod1.shift.squeeze(1).to(target_dtype)
        
        img_qkv = self.img_attn.qkv(img_modulated)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
        
        # Сборка выхлопа Double-блока (Строго 2 элемента кортежа на выходе по уставу Метрополии)
        img = img + img_mod1.gate.squeeze(1).to(target_dtype) * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate.squeeze(1).to(target_dtype) * self.img_mlp((1 + img_mod2.scale.squeeze(1).to(target_dtype)) * self.img_norm2(img) + img_mod2.shift.squeeze(1).to(target_dtype))
        
        txt = txt + txt_mod1.gate.squeeze(1).to(target_dtype) * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate.squeeze(1).to(target_dtype) * self.txt_mlp((1 + txt_mod2.scale.squeeze(1).to(target_dtype)) * self.txt_norm2(txt) + txt_mod2.shift.squeeze(1).to(target_dtype))
        return txt, img

class SingleStreamBlock(nn.Module):
    """Маршевый одиночный блок с исправленным распилом MLP Метрополии."""
    def __init__(self, hidden_size: int, num_heads: int = 24, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=torch.bfloat16)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=torch.bfloat16)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, pe: torch.Tensor, distill_vec: list, mask: torch.Tensor = None) -> torch.Tensor:
        ChromaTelemetry.verify(x, "SingleStreamBlock.x_in")
        target_dtype = self.linear1.weight.dtype
        x = x.to(target_dtype)
        mod = distill_vec
        
        x_mod = (1 + mod.scale.squeeze(1).to(target_dtype)) * self.pre_norm(x) + mod.shift.squeeze(1).to(target_dtype)
        
        # Жесткая фиксация сплита по эталонной маске кузнецов Метрополии (3*H и mlp_hidden_dim)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        attn = attention(q, k, v, pe=pe, mask=mask)
        
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + mod.gate.squeeze(1).to(target_dtype) * output
#---------------- Конец Блока №4 ----------------
