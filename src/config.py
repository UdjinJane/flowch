import os
import torch

os.environ['TORCH_CUDNN_V_GTE_9'] = '1'

DATASET_DIR = 'Z:\\flowch\\dataset\\mng_oks_bl'
METADATA_PATH = 'Z:\\flowch\\metadata.jsonl'
VAE_PATH = 'Z:\\AiModels\\models\\vae\\flux_vae.safetensors'
OUTPUT_DIR = 'Z:\\flowch'

num_epochs = 150
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 3
