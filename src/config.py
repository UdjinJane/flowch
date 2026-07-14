import os
import torch
os.environ['TORCH_CUDNN_V_GTE_9'] = '1'
DATASET_DIR = r'Z:lowch\dataset\mng_oks_bl'
METADATA_PATH = r'Z:lowch\metadata.jsonl'
VAE_PATH = r'Z:\AiModels\modelsaelux_vae.safetensors'
OUTPUT_DIR = r'Z:lowch'
num_epochs = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'
