# ИНСТРУКЦИЯ ПО ИНЖЕКЦИИ НАМЫТОГО ЗОЛОТА TORCHAO

## 1. Квантование состояний AdamW (Снижение полки VRAM на 2-3 ГБ)
Вместо стандартного прожорливого AdamW в `train_engine_v02.py` теперь можно подключить 8-битное ядро:
```python
import sys
sys.path.append("Z:\\flowch\\learn")
from extracted_optim import AdamW8bit

# Снайперская замена маршевого узла оптимизатора:
optimizer = AdamW8bit(trainable_params, lr=TrainConfig.LEARNING_RATE)
```

## 2. Агрессивный CPU-оффлоад градиентов (Полная защита от OOM при батче > 1)
```python
import torch
sys.path.append("Z:\\flowch\\learn")
from extracted_optim import CPUOffloadOptimizer

optimizer = CPUOffloadOptimizer(
    trainable_params, 
    torch.optim.AdamW, 
    lr=TrainConfig.LEARNING_RATE, 
    fused=True
)
```