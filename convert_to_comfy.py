# -*- coding: utf-8 -*-
import os
from safetensors.torch import load_file, save_file

def convert_lora_to_comfy():
    input_path = r"Z:\flowch\chroma1_mangala_lora_latent_heavy.safetensors"
    output_path = r"Z:\flowch\chroma1_mangala_lora_comfy.safetensors"
    
    if not os.path.exists(input_path):
        print(f"❌ сходный файл LoRA не найден на диске: {input_path}")
        print("одсказка: Скрипт нужно запускать только С того, как train.py полностью закончит 20 эпох!")
        return
        
    print(f"📦 Считываем кастомные веса из {os.path.basename(input_path)}...")
    src_sd = load_file(input_path)
    converted_sd = {}
    
    print("🔄 ересборка карты тензоров под стандарты ComfyUI / Kohya (Flux/Chroma)...")
    for key, tensor in src_sd.items():
        new_key = key
        
        # бработка double_blocks
        if "double_blocks" in key:
            # ревращаем 'double_blocks.0.img_attn.qkv.lora_A' 
            # в 'lora_diffusion_model_double_blocks_0_img_attn_qkv.lora_A.weight'
            new_key = key.replace("double_blocks.", "lora_diffusion_model_double_blocks_")
            new_key = new_key.replace(".img_attn.qkv.lora_", "_img_attn_qkv.lora_")
            new_key = new_key.replace(".img_attn.proj.lora_", "_img_attn_proj.lora_")
            
        # бработка single_blocks
        elif "single_blocks" in key:
            # ревращаем 'single_blocks.0.linear1.lora_A' 
            # в 'lora_diffusion_model_single_blocks_0_linear1.lora_A.weight'
            new_key = key.replace("single_blocks.", "lora_diffusion_model_single_blocks_")
            new_key = new_key.replace(".linear1.lora_", "_linear1.lora_")
            new_key = new_key.replace(".linear2.lora_", "_linear2.lora_")
            
        # обавляем обязательное для PEFT окончание весов .weight
        if not new_key.endswith(".weight"):
            new_key = new_key + ".weight"
            
        converted_sd[new_key] = tensor
        print(f"  [→] {key} \n      >>> {new_key}")
        
    save_file(converted_sd, output_path)
    print(f"\n🎉 онвертация успешно завершена! оевой файл запечен: {output_path}")
    print("го можно напрямую закидывать в папку ComfyUI/models/loras/")

if __name__ == "__main__":
    convert_lora_to_comfy()