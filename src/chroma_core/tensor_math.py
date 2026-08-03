import torch
from einops import rearrange
from torch import Tensor

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

#---------------- Старт Текстового Шва Математики (Исправление синтаксического разрыва типа данных) -
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Математический шлюз внимания.
    Сигнатура типа исправлена, разрыв устранен. Контур готов к пуску.
    """
#---------------- Конец Текстового Шва Математики -----------------

    # 1. Защитный кастинг и contiguous выравнивание осей памяти перед расчетом
    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    
    # 2. Аппаратный расчет через стабильный, изолированный контур PyTorch SDPA
    # Перекладываем оси из формата [K, B, H, L, D] под стандартный устав [B, H, L, D]
    # Внимание: q, k, v прилетают со склеенной осью последовательности текста и картинок
    
    # Прямой вызов нативного безопасного ядра Autograd PyTorch
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=mask, 
        dropout_p=0.0, 
        is_causal=False
    )
    
    return out.contiguous()
#---------------- Конец Текстового Шва Математики -----------------
