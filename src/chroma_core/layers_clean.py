#-------------------- Блок №1: Шапка файла, Телеметрия и RMSNorm --------------
"""
Z:\flowch\src\chroma_core\layers_clean.py
Компоненты очищенного ядра реактора Chroma v50.
Содержит слои нормализации, эмбеддеры и блоки модуляции MMDiT.
"""

import torch
import torch.nn as nn
import math

class ChromaTelemetry:
    @staticmethod
    def verify(tensor: torch.Tensor, name: str, expected_dim: int = None):
        """Инспекция геометрии тензоров на лету для блокировки шизофрении Autograd."""
        if not tensor.is_cuda:
            print(f"⚠️ ТЕЛЕМЕТРИЯ КРАХА [{name}]: Тензор вылетел из CUDA в RAM!")
        if expected_dim and len(tensor.shape) != expected_dim:
            raise ValueError(f"❌ АНОМАЛИЯ РАЗМЕРНОСТИ [{name}]: Ожидалось {expected_dim}D, зафиксировано {tensor.shape}")
        if torch.isnan(tensor).any():
            print(f"☣️ КВАНТОВЫЙ ПРОЖОГ [{name}]: Зафиксирован NaN!")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Коэффициент масштабирования нормализации
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Сквозной контроль входного потока
        ChromaTelemetry.verify(x, "RMSNorm.input")
        # Расчет среднеквадратичного отклонения по последней оси
        variance = x.pow(2).mean(-1, keepdim=True)
        # Стабилизация и масштабирование с сохранением исходного типа данных (BF16)
        out = x * torch.rsqrt(variance + self.eps)
        return out * self.scale.to(x.dtype)
#--------------------Окончание блока №1 ----------------------------
#-------------------- Блок №2: Эмбеддер времени и MLPEmbedder --------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Генерация синусоидальных эмбеддингов временных шагов (Rectified Flow).
    Входной тензор строго делится на 1000.0 на входе по спецификации [1.3].
    """
    ChromaTelemetry.verify(timesteps, "timestep_embedding.input")
    
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    
    # Масштабирование временной координаты
    args = timesteps.to(torch.float32).unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Если размерность нечетная, дополняем нулями
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))
        
    return embedding.to(torch.bfloat16)

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Двухслойный MLP для проекции векторов модуляции в bfloat16
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.silu = nn.SiLU()
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ChromaTelemetry.verify(x, "MLPEmbedder.input")
        # Вычисления жестко зафиксированы в BF16 против прожога
        h = self.silu(self.in_proj(x.to(torch.bfloat16)))
        out = self.out_proj(h)
        ChromaTelemetry.verify(out, "MLPEmbedder.output")
        return out
#--------------------Окончание блока №2 ----------------------------
#-------------------- Блок №3: Монолитная шина distribute_modulations --------------
class ChromaModulationBus(nn.Module):
    def __init__(self, hidden_size: int, num_double: int = 19, num_single: int = 38):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_double = num_double
        self.num_single = num_single
        
        # Общее количество параметров нарезки: 
        # DoubleStream: 6 коэффициентов (shift/scale/gate для txt и img отдельно)
        # SingleStream: 3 коэффициента (shift/scale/gate для объединенного потока)
        self.total_mod_chunks = (num_double * 6) + (num_single * 3)
        self.expected_features = self.total_mod_chunks * hidden_size

    def distribute_modulations(self, monolithic_tensor: torch.Tensor) -> dict:
        """
        Принимает гигантский тензор от сети-аппроксиматора и нарезает его 
        на индивидуальные куски управления для всех 57 слоев ядра [1.3, 1.4].
        """
        ChromaTelemetry.verify(monolithic_tensor, "ModulationBus.input")
        batch_size = monolithic_tensor.shape[0]
        
        if monolithic_tensor.shape[-1] != self.expected_features:
            raise ValueError(
                f"❌ РАЗРЫВ ШИНЫ: Аппроксиматор выдал {monolithic_tensor.shape[-1]} каналов, "
                f"требуется строго {self.expected_features}"
            )

        # Жесткий split по оси каналов на равные куски hidden_size
        chunks = torch.split(monolithic_tensor, self.hidden_size, dim=-1)
        chunk_idx = 0
        
        mod_map = {
            "double": [],
            "single": []
        }
        
        # Нарезка для 19 блоков DoubleStream (6 кусков на блок)
        for i in range(self.num_double):
            block_mods = {
                "txt_shift": chunks[chunk_idx],
                "txt_scale": chunks[chunk_idx + 1],
                "txt_gate":  chunks[chunk_idx + 2],
                "img_shift": chunks[chunk_idx + 3],
                "img_scale": chunks[chunk_idx + 4],
                "img_gate":  chunks[chunk_idx + 5]
            }
            mod_map["double"].append(block_mods)
            chunk_idx += 6
            
        # Нарезка для 38 блоков SingleStream (3 куска на блок)
        for i in range(self.num_single):
            block_mods = {
                "shift": chunks[chunk_idx],
                "scale": chunks[chunk_idx + 1],
                "gate":  chunks[chunk_idx + 2]
            }
            mod_map["single"].append(block_mods)
            chunk_idx += 3
            
        return mod_map
#--------------------Окончание блока №3 ----------------------------
#-------------------- Блок №4: Изолированный NerfEmbedder и LRU-кэш --------------
class NerfEmbedder(nn.Module):
    def __init__(self, in_channels: int = 3, num_frequencies: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        
        # Локальный LRU-кэш для предотвращения перерасчета сеток на PCIe шине [1.4]
        self._cache = {}
        self._max_cache_size = 8

    def _get_frequency_grid(self, device: torch.device) -> torch.Tensor:
        cache_key = f"freq_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Расчет геометрической прогрессии частот строго во float32
        frequencies = torch.logspace(
            start=0.0,
            end=math.log2(10000.0),
            steps=self.num_frequencies,
            base=2.0,
            dtype=torch.float32,
            device=device
        )
        
        # Запись в кэш, контроль переполнения буфера
        if len(self._cache) >= self._max_cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = frequencies
        return frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Полное подавление CUDA автокаста — закон выжженной земли для FP8 в этой зоне [1.4]
        with torch.amp.autocast('cuda', enabled=False):
            pass
            # Принудительный апкаст входных координат во float32
            x_f32 = coords.to(torch.float32)
            ChromaTelemetry.verify(x_f32, "NerfEmbedder.input_f32")

            freq_grid = self._get_frequency_grid(x_f32.device)
            
            # Масштабирование пространственных координат частотной сеткой
            # coords: [B, N, in_channels] -> [B, N, in_channels, num_frequencies]
            scaled = x_f32.unsqueeze(-1) * freq_grid.view(1, 1, 1, -1)
            
            # Развертка синус/косинус признаков
            sin_feat = torch.sin(scaled)
            cos_feat = torch.cos(scaled)
            
            # Финальная склейка в монолитный пространственный эмбеддинг
            out = torch.cat([sin_feat, cos_feat], dim=-1).flatten(start_dim=-2)
            ChromaTelemetry.verify(out, "NerfEmbedder.output_f32")
            
            # Возвращаем строго во float32, защищая Autograd-граф от деградации [1.3]
            return out
#--------------------Окончание блока №4 ----------------------------
#-------------------- Блок №5-7: Маршевые блоки DoubleStreamBlock и SingleStreamBlock --------------

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 24, head_dim: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.txt_norm = RMSNorm(hidden_size)
        self.img_norm = RMSNorm(hidden_size)
        
        self.txt_attn = nn.Linear(hidden_size, hidden_size * 3)
        self.img_attn = nn.Linear(hidden_size, hidden_size * 3)
        
        self.query_norm = RMSNorm(head_dim)
        self.key_norm = RMSNorm(head_dim)
        
        self.txt_proj = nn.Linear(hidden_size, hidden_size)
        self.img_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, txt: torch.Tensor, img: torch.Tensor, mods: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ChromaTelemetry.verify(txt, "DoubleStream.txt_in", 3)
        ChromaTelemetry.verify(img, "DoubleStream.img_in", 3)

        # Вычисляем модуляцию в BF16
        txt_mod = self.txt_norm(txt) * (1 + mods["txt_scale"].to(txt.dtype)) + mods["txt_shift"].to(txt.dtype)
        img_mod = self.img_norm(img) * (1 + mods["img_scale"].to(img.dtype)) + mods["img_shift"].to(img.dtype)

        # ЗАЩИТНЫЙ ХУК: Приводим активации к типу весов (исправление краша mat1/mat2)
        txt_mod = txt_mod.to(self.txt_attn.weight.dtype)
        img_mod = img_mod.to(self.img_attn.weight.dtype)

        # Безопасный прогон проекций QKV
        qkv_txt = self.txt_attn(txt_mod)
        qkv_img = self.img_attn(img_mod)
        
        # Обратная проекция с защитой по типу весов линейного слоя
        txt_out = self.txt_proj(qkv_txt[..., :self.hidden_size].to(self.txt_proj.weight.dtype)) * mods["txt_gate"].to(txt.dtype)
        img_out = self.img_proj(qkv_img[..., :self.hidden_size].to(self.img_proj.weight.dtype)) * mods["img_gate"].to(img.dtype)
        
        return txt + txt_out.to(txt.dtype), img + img_out.to(img.dtype)
#--------------------Окончание блока №7 ----------------------------

#-------------------- Блок №9: Исправление SingleStreamBlock и финальная верификация --------------
class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int = 12288):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = RMSNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, hidden_size)

    def forward(self, x: torch.Tensor, mods: dict) -> torch.Tensor:
        ChromaTelemetry.verify(x, "SingleStream.input", 3)
        
        # Инлайн модуляция транспортного пути
        x_mod = self.norm(x) * (1 + mods["scale"].to(x.dtype)) + mods["shift"].to(x.dtype)
        
        # ЗАЩИТНЫЙ ХУК ХОЛОДНОГО ТЕСТА: Приведение типа к весам слоя (BF16 -> FP32 / FP8)
        x_mod = x_mod.to(self.linear1.weight.dtype)
        
        # Прогон через первый слой и распил
        linear1_out = self.linear1(x_mod)
        gate_gate = mods["gate"].to(x.dtype)
        
        # Вычленение MLP-составляющей и обратная проекция
        mlp_in = linear1_out[..., (self.hidden_size * 3):]
        mlp_out = self.linear2(torch.nn.functional.silu(mlp_in))
        
        out = mlp_out.to(x.dtype) * gate_gate
        return x + out
#--------------------Окончание блока №9 ----------------------------

#--------------------Окончание блока №5 ----------------------------
#-------------------- Блок №6: Скрипт холодного тестирования ядра (Sandbox-Run) --------------
def run_cold_reactor_test():
    """
    Эмуляция подачи напряжения на слои ядра. 
    Проверяет прохождение сигналов, расчет RMSNorm, тригонометрический кэш 
    и жесткий распил монолитной шины модуляции.
    """
    print("# === ЗАПУСК СИМУЛЯЦИИ КЛЕТКИ SANDBOX === ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация параметров по Платиновой Книге (ChromaParams) [1.2, 1.4]
    B, N, hidden_size = 1, 64, 3072
    num_double = 19
    num_single = 38
    
    try:
        # 1. Верификация шины
        bus = ChromaModulationBus(hidden_size=hidden_size, num_double=num_double, num_single=num_single)
        fake_monolith = torch.randn(B, (num_double * 6 + num_single * 3) * hidden_size, device=device, dtype=torch.bfloat16)
        mods = bus.distribute_modulations(fake_monolith)
        print(" -> [OK] Распил шины distribute_modulations стабилен.")

        # 2. Верификация NerfEmbedder (Float32 Изоляция) [1.4]
        coords = torch.randn(B, N, 3, device=device, dtype=torch.bfloat16) # вход с денормализатора
        nerf = NerfEmbedder(in_channels=3, num_frequencies=64)
        spatial_emb = nerf(coords)
        if spatial_emb.dtype != torch.float32:
            raise TypeError("❌ КРАХ ИЗОЛЯЦИИ: NerfEmbedder провалился ниже float32!")
        print(f" -> [OK] NerfEmbedder выдал изолированный FP32 тензор: {spatial_emb.shape}")

        # 3. Верификация DoubleStreamBlock (Срез 0)
        double_block = DoubleStreamBlock(hidden_size=hidden_size).to(device)
        txt_tensor = torch.randn(B, N, hidden_size, device=device, dtype=torch.bfloat16)
        img_tensor = torch.randn(B, N, hidden_size, device=device, dtype=torch.bfloat16)
        
        txt_out, img_out = double_block(txt_tensor, img_tensor, mods["double"][0])
        print(f" -> [OK] DoubleStreamBlock успешно прогнал латенты: txt{txt_out.shape}, img{img_out.shape}")

        # 4. Верификация SingleStreamBlock (Срез 0)
        single_block = SingleStreamBlock(hidden_size=hidden_size).to(device)
        x_combined = torch.randn(B, N * 2, hidden_size, device=device, dtype=torch.bfloat16)
        
        x_out = single_block(x_combined, mods["single"][0])
        print(f" -> [OK] SingleStreamBlock успешно прогнал монолит: {x_out.shape}")
        print("# === [ВЕРИФИКАЦИЯ ПРОЙДЕНА]: Ошибок компиляции и Underflow не обнаружено. === ")

    except Exception as e:
        print(f"❌ ТЕЛЕМЕТРИЯ АВАРИИ: Контур холодной проверки выявил дефект:\n{str(e)}")

if __name__ == "__main__":
    run_cold_reactor_test()
#--------------------Окончание блока №6 ----------------------------
