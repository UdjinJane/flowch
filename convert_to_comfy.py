import os
import torch
from safetensors.torch import load_file, save_file
from src.config import OUTPUT_DIR

def convert_raw_lora_to_comfy():
    print('📦 нициализация конвертера весов...')
    raw_lora_path = os.path.join(OUTPUT_DIR, 'chroma1_mangala_lora.safetensors')
    comfy_lora_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'chroma1_mangala_comfy.safetensors')
    
    if not os.path.exists(raw_lora_path):
        print(f'⚠ Сырые веса не найдены: {raw_lora_path}. Сначала запустите тренировку!')
        return
        
    print(f'🔄 агрузка сырых матриц: {raw_lora_path}')
    state_dict = load_file(raw_lora_path)
    comfy_dict = {}
    
    for key, value in state_dict.items():
        comfy_key = key.replace('_lora_A', '.lora_down.weight').replace('_lora_B', '.lora_up.weight')
        comfy_dict[comfy_key] = value.to(torch.float16)
        
    os.makedirs(os.path.dirname(comfy_lora_path), exist_ok=True)
    save_file(comfy_dict, comfy_lora_path)
    print(f'🎉 онвертация завершена! отово для ComfyUI: {comfy_lora_path}')

if __name__ == '__main__':
    convert_raw_lora_to_comfy()
