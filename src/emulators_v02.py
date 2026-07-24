import torch
import torch.nn as nn

class ChromaTransformerMock(nn.Module):
    """ Эмулятор маршевого трансформера Chroma1 (FP8-Scaled) """
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.get("num_layers", 19)
        self.channels = 64  # Жесткая размерность каналов траектории из transformer_config.json
        
    def forward(self, hidden_states, timestep, text_embeddings, pooled_projections, txt_ids, img_ids, text_ids_mask=None):
        # Прецизионный аудит входящих осей
        assert hidden_states.ndim == 3, f"[КРАХ ЭМУЛЯТОРА] Ожидалась 3D геометрия кадра, прилетело: {hidden_states.shape}"
        assert hidden_states.shape[2] == self.channels, f"[КРАХ ЭМУЛЯТОРА] Размерность каналов {hidden_states.shape[2]} не равна {self.channels}"
        
        batch = hidden_states.shape[0]
        img_len = hidden_states.shape[1]      # 1024 патча кадра
        txt_len = text_embeddings.shape[1]   # 256 токенов текста
        total_seq = img_len + txt_len        # Общая длина последовательности = 1280
        
        # Возвращаем идеальный объединенный вектор скорости (B, 1280, 64) под наш снайперский срез
        return torch.zeros((batch, total_seq, self.channels), dtype=hidden_states.dtype, device=hidden_states.device)

class ChromaVAEMock(nn.Module):
    """ Эмулятор VAE-декодера """
    def __init__(self, config_dict=None):
        super().__init__()
        
    def decode(self, latents, return_dict=True):
        # Латенты (1, 64, 32, 32) -> Имитируем выхлоп кадра манги (1, 3, 256, 256)
        batch = latents.shape[0]
        mock_image = torch.zeros((batch, 3, 256, 256), dtype=latents.dtype, device=latents.device)
        
        # Контракт Diffusers ожидает объект с кортежем или атрибутом sample
        class VAEDecoderOutput:
            def __init__(self, sample): 
                self.sample = sample
            def __getitem__(self, index):
                if index == 0: return self.sample
                raise IndexError
        return VAEDecoderOutput(sample=mock_image)
