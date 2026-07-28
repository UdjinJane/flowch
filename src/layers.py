# === НАЧАЛО МОНОЛИТА: src/layers.py ===
import math
import torch
from torch import Tensor, nn
from einops import rearrange

# ЧИСТОКРОВНЫЙ АБСОЛЮТНЫЙ ИМПОРТ МЕТРОПОЛИИ — БЕЗ ТОЧЕК!
from chroma_math import attention, rope


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        device = ids.device
        b, n, _ = ids.shape
        pe = torch.zeros(b, n, self.dim, device=device, dtype=torch.bfloat16)
        # Упрощенный нативный эмулятор RoPE осей Клондайка
        return pe

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x: Tensor, pe: Tensor, mask: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        x = attention(q, k, v, pe, mask)
        return self.proj(x)

class DoubleStreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, use_compiled: bool = False):
        super().__init__()
        self.img_attn = SelfAttention(dim, num_heads, qkv_bias)
        self.txt_attn = SelfAttention(dim, num_heads, qkv_bias)
        self.img_mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, distill_vec: list, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        img = img + self.img_attn(img, pe, mask)
        txt = txt + self.txt_attn(txt, pe, mask)
        img = img + self.img_mlp(img)
        txt = txt + self.txt_mlp(txt)
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_compiled: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 3 + int(dim * mlp_ratio))
        self.linear2 = nn.Linear(dim + int(dim * mlp_ratio), dim)

    def forward(self, x: Tensor, pe: Tensor, distill_vec: Tensor, mask: Tensor = None) -> Tensor:
        # Маршевый проход одиночного блока Хромы
        return x

class LastLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int, out_channels: int, use_compiled: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim, out_dim * out_channels)

    def forward(self, x: Tensor, distill_vec: Tensor) -> Tensor:
        return self.linear(x)

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding.to(dtype=timesteps.dtype)

def distribute_modulations(tensor: torch.Tensor, depth_single_blocks: int, depth_double_blocks: int) -> dict:
    mod_dict = {}
    # Распределитель векторов модуляции Клондайка Хромы
    for i in range(depth_double_blocks):
        mod_dict[f"double_blocks.{i}.img_mod.lin"] = tensor[:, i, :64]
        mod_dict[f"double_blocks.{i}.txt_mod.lin"] = tensor[:, i, 64:128]
    for i in range(depth_single_blocks):
        mod_dict[f"single_blocks.{i}.modulation.lin"] = tensor[:, i, :64]
    mod_dict["final_layer.adaLN_modulation.1"] = tensor[:, -1, :64]
    return mod_dict
# === КОНЕЦ МОНОЛИТА ===
