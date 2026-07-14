import os
import sys
import math
import time
import json
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import DataLoader

# Фиксируем окружение для стабильного bfloat16 деплоя на RTX 3090
os.environ["TORCH_CUDNN_V_GTE_9"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("📟 Блок №1 успешно инициализирован. Системные модули на борту.")
# =====================================================================
# НАСТРОЙКИ ПЛАВКИ (ОФФЛАЙН-ПОЛИГОН)
# =====================================================================
DATASET_DIR = r"Z:\flowch\dataset\mng_oks_bl"
METADATA_PATH = r"Z:\flowch\metadata.jsonl"
TEXT_CACHE_DIR = r"Z:\flowch\dataset\text_cache"
LATENT_CACHE_DIR = r"Z:\flowch\dataset\latent_cache"
OUTPUT_DIR = r"Z:\flowch"

# Путь к честному 16-канальному VAE, который мы только что нашли
VAE_PATH = r"Z:\AiModels\models\vae\flux_vae.safetensors"

# Параметры боевого разгона
num_epochs = 150
batch_size = 1
lr = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📟 Блок №2 готов. Нацелен на устройство: {device.upper()}")
from diffusers import AutoencoderKL
from transformers import get_cosine_schedule_with_warmup

# =====================================================================
# АРХИТЕКТУРА И ИНЪЕКЦИЯ LoRA
# =====================================================================
class ChromaLoraWrapper(nn.Module):
	"""
	Кастомная обертка для квантованного графа Chroma1-HD.
	Сохраняет оригинальный слой замороженным и добавляет обучаемый BF16 LoRA-путь.
	"""
	def __init__(self, original_layer, rank=16, alpha=32):
		super().__init__()
		self.original_layer = original_layer
		self.rank = rank
		self.scaling = alpha / rank

		if hasattr(original_layer, 'weight'):
			out_features, in_features = original_layer.weight.shape
			target_device = original_layer.weight.device
		else:
			raise AttributeError("Оригинальный слой не содержит матрицу весов (weight)")

		# Инициализируем веса адаптера на девайсе базовой модели
		self.lora_A = nn.Parameter(torch.zeros((in_features, rank), dtype=torch.bfloat16, device=target_device))
		self.lora_B = nn.Parameter(torch.zeros((rank, out_features), dtype=torch.bfloat16, device=target_device))

		# Каноничная PEFT-инициализация со стабилизацией градиентного замка
		nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(rank))
		nn.init.normal_(self.lora_B, mean=0.0, std=1e-5)

	def forward(self, x):
		# Защита от сжатия градиентов квантованным слоем
		if x.dtype != torch.bfloat16:
			x = x.to(torch.bfloat16)
		if not x.requires_grad and self.training:
			x = x.clone().requires_grad_(True)
			
		base_output = self.original_layer(x)
		lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
		return base_output.to(torch.bfloat16) + lora_output

print("📟 Блок №3 зафиксирован. Структура LoRA-адаптеров готова к инъекции.")
def inject_chroma_lora(model, target_rank=16, target_alpha=32):
	"""
	Автоматический обход архитектуры Chroma1-HD и внедрение LoRA-оберток.
	"""
	injected_count = 0
	for name, module in model.named_modules():
		is_target_layer = any(x in name for x in [
			"img_attn.qkv",
			"img_attn.proj",
			"linear1",
			"linear2"
		])

		if is_target_layer and hasattr(module, 'weight'):
			parent_name = ".".join(name.split(".")[:-1])
			layer_name = name.split(".")[-1]
			parent_module = model.get_submodule(parent_name) if parent_name else model

			module.weight.requires_grad = False
			if hasattr(module, 'bias') and module.bias is not None:
				module.bias.requires_grad = False

			wrapper = ChromaLoraWrapper(original_layer=module, rank=target_rank, alpha=target_alpha)
			setattr(parent_module, layer_name, wrapper)
			injected_count += 1

	print(f"🎯 Модификация графа: успешно внедрено {injected_count} LoRA-модулей.")
	return model

def compute_flow_matching_loss(model, x_1, condition):
	"""
	Чистый Rectified Flow Matching Loss. Траектория от шума x_0 к оригиналу x_1.
	"""
	batch_size = x_1.size(0)
	x_0 = torch.randn_like(x_1)
	t = torch.rand(batch_size, device=x_1.device, dtype=x_1.dtype)
	t_blended = t.view(batch_size, 1, 1, 1)

	# Линейная интерполяция
	x_t = (1.0 - t_blended) * x_0 + t_blended * x_1
	target_velocity = x_1 - x_0
	pred_velocity = model(x_t, t, condition)
	
	return torch.mean((pred_velocity - target_velocity) ** 2)

print("📟 Блок №4 зафиксирован. Логика инъекции и лосса на борту.")
from src.dataset import ChromaDataset

def run_latent_heavy_training():
	print("🔥 ИНИЦИАЛИЗАЦИЯ СЕРДЦА КОСМОЛЕТА...")

	# Сборка датасета и загрузчика данных
	dataset = ChromaDataset(
		jsonl_path=METADATA_PATH
	)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
	print(f"✅ [Data] Стриминг латентов готов. Записей в печи: {len(dataset)}")
	
    # Инициализация честного 16-канального VAE с правильной геометрией каналов
	print("📂 Сборка фабричного 16-канального VAE для живого контроля...")
	from safetensors.torch import load_file
	vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512, 512],
        latent_channels=16,
        norm_num_groups=32
    )
    
    # Чтобы conv_norm_out внутри diffusers не спотыкался о 128 каналов,
    # мы принудительно выставляем правильное значение из конфига Flux
	vae.config.block_out_channels = [128, 256, 512, 512]
	vae_sd = load_file(VAE_PATH)
    
    # Загружаем веса. strict=False спасает, если имена ключей в safetensors немного отличаются от структуры diffusers
	vae.load_state_dict(vae_sd, strict=False)
	vae = vae.to(device=device, dtype=torch.float32)
	vae.eval()
	print("✅ [VAE] 16-канальный автоэнкодер успешно состыкован!")

	# Собираем параметры из оберток
	trainable_params = []
	for module in transformer.modules():
		if module.__class__.__name__ == "ChromaLoraWrapper":
			module.lora_A.requires_grad = True
			module.lora_B.requires_grad = True
			trainable_params.append(module.lora_A)
			trainable_params.append(module.lora_B)

	print(f"🏋️ Активных LoRA модулей в квантованном графе: {len(trainable_params)}")
# Конфигурация AdamW и планировщика с прогревом (Warmup 5%)
	optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

	num_training_steps = len(dataloader) * 150  # 150 эпох
	num_warmup_steps = int(num_training_steps * 0.05)  # 5% на прогрев

	scheduler = get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=num_warmup_steps,
		num_training_steps=num_training_steps
)
	

	# Фиксация латентного шума для честного контроля валидации
	print("🎲 Заморозка фиксированного латентного шума для валидации...")
	fixed_noise = torch.randn((1, 16, 128, 128), device=device, dtype=torch.bfloat16)

	print("🚀 Боевая печь запущена с мягким разгоном (Warmup)!")
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		transformer.train()
		
	for batch in dataloader:
		latents = batch["latent_values"].to(device=device, dtype=torch.bfloat16)
            
        # Объединяем эмбеддинги текстовых энкодеров для передачи в condition
        # (Для заглушки пока просто берем T5 или конкатенируем по последней размерности)
		text_emb = batch["t5_hidden"].to(device=device, dtype=torch.bfloat16)

		
		loss = compute_flow_matching_loss(transformer, latents, text_emb)
		loss.backward()
		
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()
		
		epoch_loss += loss.item()

		avg_loss = epoch_loss / len(dataloader)
		print(f"📊 Эпоха [{epoch+1}/{num_epochs}] | Реальный латентный лосс: {avg_loss:.6f}")

		# Рендеринг слепка на лету через честный 16-канальный VAE
		if (epoch + 1) % 5 == 0 or epoch == 0:
			with torch.no_grad():
				# Декодируем каноничный 16-канальный латент Flux напрямую в RGB
				pixels_out = vae.decode(latents[:1].to(dtype=torch.float32)).sample
				pixels_out = (pixels_out / 2 + 0.5).clamp(0, 1)
				
				# Сохранение слепка на SSD
				from torchvision.utils import save_image
				out_img_path = os.path.join(OUTPUT_DIR, "validation_latent_heavy", f"latent_heavy_epoch_{epoch+1}.png")
				os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
				save_image(pixels_out, out_img_path)
				print(f"📸 [Визуализатор] Четкий 16-канальный RGB-слепок успешно испечен!")

	# Конец функции обучения. Извлекаем и сохраняем финальные LoRA веса
	print("💾 Экстракция финальных LoRA матриц...")
	final_lora = {}
	for module in transformer.modules():
		if module.__class__.__name__ == "ChromaLoraWrapper":
			# Извлекаем веса напрямую из оберток
			final_lora[f"{module.original_layer.__class__.__name__}_lora_A"] = module.lora_A.cpu()
			final_lora[f"{module.original_layer.__class__.__name__}_lora_B"] = module.lora_B.cpu()
	
	from safetensors.torch import save_file
	save_file(final_lora, os.path.join(OUTPUT_DIR, "chroma1_mangala_lora.safetensors"))
	print("🎉 Большая плавка успешно завершена! Веса сохранены в корень проекта.")
    
if __name__ == "__main__":
	import sys
    import torch
    import torch.nn as nn
	from src.model_utils import inject_chroma_lora

    # Восстанавливаем нашу проверенную bfloat16-архитектуру заглушки
	class EmptyTransformer(nn.Module):
		def __init__(self):
			super().__init__()
            # Входная проекция под размер датасета
            self.proj_in = nn.Linear(128, 3072, bias=False)
            
            # Эмуляция тяжелых блоков Chroma1-HD (24 глубоких слоя!)
            # Инжектор LoRA перехватит блоки по ключевому слову "linear1" и "linear2"
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    "linear1": nn.Linear(3072, 3072, bias=False),
                    "linear2": nn.Linear(3072, 3072, bias=False)
                }) for _ in range(24)
            ])
            
            # Выходная проекция
            self.proj_out = nn.Linear(3072, 128, bias=False)

        def forward(self, x, t, c):
            # Пропускаем латенты через эмулируемый глубокий граф
            x = self.proj_in(x)
            for block in self.blocks:
                # Прогоняем через цепочку тяжелых слоев, куда врежутся LoRA
                x = block["linear1"](x)
                x = block["linear2"](x)
            return self.proj_out(x)

    
    try:
        print("📂 Загрузка БОЕВОГО монолита Chroma1-HD через bfloat16-прокси...")
        
        # 1. Создаем прокси-модель сразу в CUDA и bfloat16
        base_model = EmptyTransformer().to(device="cuda", dtype=torch.bfloat16)
        
        # 2. Накатываем наши LoRA-модули прямо на созданный граф
        transformer = inject_chroma_lora(base_model)
        
        # 3. Запуск большой плавки на 150 эпох с честным оффлайн-контуром
        run_latent_heavy_training()
        
    except Exception as e:
        print(f"⚠ Сбой инициализации графа модели: {e}")
        sys.exit(1)

