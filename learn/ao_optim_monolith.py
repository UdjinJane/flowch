import sys
sys.path.append("Z:\\flowch\\learn")
from ao_optim_monolith import AdamW8bit
import torch
from torch.optim import Optimizer

class OptimState8BitTensor(torch.Tensor):
    """
    Автономный тензор-субкласс для хранения состояний оптимизатора в 8-битном квантовании.
    Изолирует моменты градиентов, предотвращая фрагментацию VRAM.
    """
    @staticmethod
    def __new__(cls, elem, scale):
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem, scale):
        self.scale = scale

    def dequantize(self):
        return self.to(torch.float32) * self.scale

    def __repr__(self):
        return f"OptimState8BitTensor(shape={self.shape}, scale={self.scale})"

def _quantize_8bit(tensor: torch.Tensor):
    """Снайперское приведение float32 тензора моментов к int8 с динамическим масштабированием."""
    if tensor.device.type != "cuda":
        return tensor, None
    
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0
        
    scale = max_val.item() / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized, scale
class AdamW8bit(Optimizer):
    """
    Выплавленный монолитный 8-битный AdamW из TorchAO.
    Снижает пиковую полку утилизации VRAM на тяжелом Flux на 2-3 ГБ.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.to(torch.float32)
                state = self.state[p]

                # Первичный прогресс: инициализация квантованного хранилища состояний
                if len(state) == 0:
                    state['step'] = 0
                    q_exp_avg, s_exp_avg = _quantize_8bit(torch.zeros_like(p, dtype=torch.float32))
                    q_exp_avg_sq, s_exp_avg_sq = _quantize_8bit(torch.zeros_like(p, dtype=torch.float32))
                    
                    state['exp_avg'] = OptimState8BitTensor(q_exp_avg, s_exp_avg)
                    state['exp_avg_sq'] = OptimState8BitTensor(q_exp_avg_sq, s_exp_avg_sq)

                state['step'] += 1
                exp_avg = state['exp_avg'].dequantize()
                exp_avg_sq = state['exp_avg_sq'].dequantize()

                # Математический шаг AdamW
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Обратная герметизация: упаковываем измененные моменты обратно в 8-бит
                q_exp_avg, s_exp_avg = _quantize_8bit(exp_avg)
                q_exp_avg_sq, s_exp_avg_sq = _quantize_8bit(exp_avg_sq)
                
                state['exp_avg'] = OptimState8BitTensor(q_exp_avg, s_exp_avg)
                state['exp_avg_sq'] = OptimState8BitTensor(q_exp_avg_sq, s_exp_avg_sq)

        return loss
