from dataclasses import dataclass
import torch
from torch import Tensor, nn
from chroma_math import apply_rope


# ЧИСТОКРОВНЫЙ АБСОЛЮТНЫЙ ИМПОРТ МЕТРОПОЛИИ — ТОЧКИ ВЫЖЖЕНЫ НАВСЕГДА!
from layers import (
    DoubleStreamBlock, EmbedND, LastLayer, SingleStreamBlock,
    timestep_embedding, Approximator, distribute_modulations
)

@dataclass
class ChromaParams:
    in_channels: int = 64
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list = None
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed_dim: int = 256
    distilled_guidance_layer: int = 11

# ЭТАЛОННАЯ ТОПОЛОГИЯ ОСЕЙ ROPE КЛОНДАЙКА
chroma_params = ChromaParams(axes_dim=[16, 56, 56])


def modify_mask_to_attend_padding(mask: Tensor, max_seq_length: int, num_extra_padding: int = 8) -> Tensor:
    """Удерживает паддинг-токены текстового процессора T5XXL в маске."""
    if mask is None:
        return None
    b, n = mask.shape
    if n <= max_seq_length:
        return mask
    modified_mask = mask.clone()
    modified_mask[:, max_seq_length : max_seq_length + num_extra_padding] = 1
    return modified_mask

class Chroma(nn.Module):
    def __init__(self, params: ChromaParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.hidden_size = params.hidden_size
        
        # Входные проекторы кадра и текста Клондайка Хромы
        self.img_in = nn.Linear(params.in_channels, params.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size)
        
        # Позиционный радар RoPE
        self.pe_embedder = EmbedND(dim=params.hidden_size, theta=params.theta, axes_dim=params.axes_dim)
        
        # Магистральные блоки обработки мантиссы
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(params.hidden_size, params.num_heads, params.mlp_ratio, params.qkv_bias)
            for _ in range(params.depth)
        ])
        
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(params.hidden_size, params.num_heads, params.mlp_ratio)
            for _ in range(params.depth_single_blocks)
        ])
        
        # Векторный дистиллятор моб-векторов
        self.guidance_in = nn.Sequential(
            nn.Linear(1, params.guidance_embed_dim),
            nn.GELU(),
            nn.Linear(params.guidance_embed_dim, params.guidance_embed_dim)
        )
        
        # Рассчитываем суммарную длину вектора модуляции по формуле Метрополии:
        # 3 вектора на каждый одиночный блок + 2 * 6 векторов экспертов на двойные блоки + 2 на финал
        total_mod_elements = 3 * params.depth_single_blocks + 12 * params.depth + 2
        self.approximator = Approximator(
            in_dim=params.guidance_embed_dim + params.hidden_size,
            out_dim=total_mod_elements * 64,
            hidden_dim=1024,
            depth=4
        )
        
        # ТОЧНАЯ КАЛИБРОВКА РАЗМЕРНОСТИ ПОД ГЕОМЕТРИЮ ЧЕКПОИНТА [64, 3072]
        self.final_layer = LastLayer(params.hidden_size, out_dim=1, out_channels=64)


    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, 
                txt_mask: Tensor, timesteps: Tensor, guidance: Tensor, 
                attn_padding: int = 1) -> Tensor:
        
        # 1. Проекция в скрытое пространство
        img = self.img_in(img)
        txt = self.txt_in(txt)
        
        # 2. Вычисление позиционных эмбеддингов
        pe_img = self.pe_embedder(img_ids)
        pe_txt = self.pe_embedder(txt_ids)
        pe = torch.cat([pe_txt, pe_img], dim=1)
        
        # 3. Инжекция вектора дистилляции и генерация моб-матриц
        vec = self.guidance_in(guidance.unsqueeze(-1).to(dtype=img.dtype))
        vec_mixed = torch.cat([vec, txt.mean(dim=1)], dim=-1)
        modulations = self.approximator(vec_mixed).view(img.shape[0], -1, 64)
        mod_dict = distribute_modulations(modulations, self.params.depth_single_blocks, self.params.depth)
        
        # Выравнивание маски под объединенный поток
        if txt_mask is not None:
            txt_mask = modify_mask_to_attend_padding(txt_mask, max_seq_length=128, num_extra_padding=attn_padding)
            img_mask = torch.ones(img.shape[0], img.shape[1], device=img.device, dtype=img.dtype)
            full_mask = torch.cat([txt_mask, img_mask], dim=1)
        else:
            full_mask = None

        # 4. Прогон через DoubleStream блоки
        for i, block in enumerate(self.double_blocks):
            distill_vec = [mod_dict[f"double_blocks.{i}.img_mod.lin"], mod_dict[f"double_blocks.{i}.txt_mod.lin"]]
            img, txt = block(img, txt, pe, distill_vec, full_mask)
            
        # 5. Объединение потоков и прогон через SingleStream блоки
        x = torch.cat([txt, img], dim=1)
        for i, block in enumerate(self.single_blocks):
            distill_vec = mod_dict[f"single_blocks.{i}.modulation.lin"]
            x = block(x, pe, distill_vec, full_mask)
            
        # Нарезаем обратно и забираем кадр
        img = x[:, txt.shape[1]:]
        
        # 6. Финальный выходной слой
        final_vec = mod_dict["final_layer.adaLN_modulation.1"]
        return self.final_layer(img, final_vec)
