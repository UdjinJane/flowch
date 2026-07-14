import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from safetensors.torch import load_file
from src.config import VAE_PATH, device

class EmptyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(128, 3072, bias=False)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'linear1': nn.Linear(3072, 3072, bias=False),
                'linear2': nn.Linear(3072, 3072, bias=False)
            }) for _ in range(24)
        ])
        self.proj_out = nn.Linear(3072, 128, bias=False)

    def forward(self, x, t, c):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block['linear1'](x)
            x = block['linear2'](x)
        return self.proj_out(x)

def build_vae():
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=['DownEncoderBlock2D']*4,
        up_block_types=['UpDecoderBlock2D']*4,
        block_out_channels=[128, 256, 512, 512],
        latent_channels=16, norm_num_groups=32
    )
    vae.load_state_dict(load_file(VAE_PATH), strict=False)
    return vae.to(device=device, dtype=torch.float32).eval()
