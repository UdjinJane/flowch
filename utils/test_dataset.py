mport sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.dataset import ChromaDataset

try:
    dataset = ChromaDataset(jsonl_path=r"Z:\flowch\metadata.jsonl")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("🔄 роверка чтения латентного батча...")
    batch = next(iter(dataloader))
    
    print("\n🚀 атентный конвейер данных полностью исправен!")
    print(f"  - атентные веса VAE: {batch['latent_values'].shape} | Dtype: {batch['latent_values'].dtype}")
    print(f"  - CLIP-L кэш: {batch['clip_hidden'].shape} | Dtype: {batch['clip_hidden'].dtype}")
    print(f"  - T5-XXL кэш: {batch['t5_hidden'].shape} | Dtype: {batch['t5_hidden'].dtype}")
    print(f"  - адры: {batch['img_name']}")
    
except Exception as e:
    print(f"❌ Сбой теста: {e}")