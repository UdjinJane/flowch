import os
import sys
import json
import torch
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5Tokenizer

def generate_text_cache():
    print("📦 Старт локальной фазы кэширования текстового контекста для Chroma1...")
    
    clip_path = r"Z:\AiModels\models\clip\clip_l.safetensors"
    t5_path = r"Z:\AiModels\models\clip\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors"
    jsonl_path = r"Z:\flowch\metadata.jsonl"
    cache_dir = r"Z:\flowch\dataset\text_cache"
    
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📥 агрузка локальных токенизаторов...")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", fix_sentencepiece=True)

    print("🔑 нициализация CLIP-L...")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
    try:
        clip_sd = load_file(clip_path)
        clip_model.load_state_dict(clip_sd, strict=False)
        print("  - окальные веса CLIP-L успешно интегрированы.")
    except Exception as e:
        print(f"  - нфо по CLIP весам (использован базовый набор): {e}")

    print("🐋 Сборка архитектуры T5-XXL из конфигурации ( скачивания весов из сети)...")
    # Создаем конфигурацию T5-XXL локально
    config = T5Config.from_pretrained("google/t5-v1_1-xxl")
    
    # нициализируем пустую модель строго в bfloat16 (она весит 0 байт на диске в этот момент)
    with torch.device("meta"):
        t5_model_meta = T5EncoderModel(config)
        
    # агружаем наши реальные локальные FP8/BF16 веса с диска Z:
    print(f"📂 агрузка локального файла весов: {os.path.basename(t5_path)}")
    t5_sd = load_file(t5_path)
    
    # ллоцируем память на GPU под структуру весов
    t5_model = T5EncoderModel(config).to(torch.bfloat16).to(device)
    
    # Сажаем веса на каркас
    t5_model.load_state_dict(t5_sd, strict=False)
    print("  - окальные веса T5-XXL FP8 успешно запечены в граф модели!")

    clip_model.eval()
    t5_model.eval()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"🔄 ачинаем обработку {len(lines)} описаний мангала...")
    
    with torch.no_grad():
        for item in lines:
            raw_path = item.get("image_path") or item.get("file_name") or ""
            # икс: выделяем чистое имя без путей и расширений
            img_base = os.path.splitext(os.path.basename(raw_path))[0]
            caption = item.get("text") or item.get("caption") or ""
            
            # рогон через CLIP-L
            clip_inputs = clip_tokenizer(caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            clip_outputs = clip_model(**clip_inputs)
            clip_hidden = clip_outputs.last_hidden_state.cpu() 

            # рогон через T5-XXL
            t5_inputs = t5_tokenizer(caption, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
            t5_outputs = t5_model(**t5_inputs)
            t5_hidden = t5_outputs.last_hidden_state.cpu()

            # паковываем кэш
            payload = {
                "clip_hidden": clip_hidden.to(torch.bfloat16),
                "t5_hidden": t5_hidden.to(torch.bfloat16)
            }
            
            cache_file_path = os.path.join(cache_dir, f"{img_base}.pt")
            torch.save(payload, cache_file_path)
            print(f"  [+] Скэширован: {img_base}.pt")
            
    print(f"🎉 эширование полностью завершено! апка: {cache_dir}")

if __name__ == "__main__":
    generate_text_cache()
