import torch
from config import TrainConfig

class FluxFlowMathV01:
    @staticmethod
    def apply_timestep_shift(t, shift=3.0):
        # анонический нелинейный сдвиг временной шкалы ядра FLUX
        # Смещает плотность зашумления к центру, повышая точность проработки деталей стали
        return (shift * t) / (1.0 + (shift - 1.0) * t)

    @classmethod
    def generate_train_timesteps(cls, batch_size, device):
        # енерируем случайное базовое время t от 0 до 1 для каждого кадра в батче
        t_base = torch.rand((batch_size,), device=device, dtype=torch.bfloat16)
        
        # акатываем наш жесткий нелинейный шифтинг
        t_shifted = cls.apply_timestep_shift(t_base, shift=3.0)
        
        # азворачиваем тензор времени до 4D размерности латентов [B, 1, 1, 1] для умножения
        return t_shifted.view(-1, 1, 1, 1)

    @staticmethod
    def blend_noise_and_latents(latents, noise, t):
        # инейная интерполяция Rectified Flow (Flow Matching)
        # Соединяет чистый кадр мангала и случайный гауссов шум по кратчайшей прямой
        noisy_latents = (1.0 - t) * latents + t * noise
        
        # ычисляем целевой вектор скорости изменения (Target Flow)
        # менно эту разницу LoRA адаптеры обязаны научиться предсказывать
        target_flow = noise - latents
        
        return noisy_latents, target_flow

if __name__ == "__main__":
    # ыстрый приборный Т-тест изоляции математического отсека
    print("[Т] Тестирование математического ядра: flow_math_v01")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # оделируем фейковые латенты нашего батча
    fake_latents = torch.randn((1, 16, 16, 16), device=device, dtype=torch.bfloat16)
    fake_noise = torch.randn_like(fake_latents)
    
    # рогон через генератор времени и блендер шума
    t = FluxFlowMathV01.generate_train_timesteps(1, device)
    noisy_x, target = FluxFlowMathV01.blend_noise_and_latents(fake_latents, fake_noise, t)
    
    print("--- [Т] ТТС ТСТ V01 СТЬ  ---")
    print(f"ектор времени t (сдвинутый): {t.item():.4f}")
    print(f"орма зашумленного латента:  {noisy_x.shape}")
    print(f"орма целевого вектора loss: {target.shape}")
