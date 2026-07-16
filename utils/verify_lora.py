import safetensors.torch
import os

path = r'Z:\flowch\checkpoints\chroma1_mangala_lora_epoch_5.safetensors'
if not os.path.exists(path):
    print('❌ Файл не найден!')
else:
    try:
        tensors = safetensors.torch.load_file(path)
        print(f'✅ Файл успешно прочитан! Всего тензоров: {len(tensors)}')
        print('\n📋 Первые 5 ключей графа весов:')
        for i, k in enumerate(list(tensors.keys())[:5]):
            print(f'  {i+1}. {k} -> shape: {list(tensors[k].shape)}')
    except Exception as e:
        print(f'❌ Ошибка чтения: {e}')
