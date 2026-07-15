# === БЛОК 2 СТАРТ ===
def main_train_loop():
    print("[ОБТ] Шаг 5.5: Запуск финального экономного диспетчера: train_engine_v02")
    
    if not os.path.exists(TrainConfig.OUTPUT_DIR):
        os.makedirs(TrainConfig.OUTPUT_DIR)
        
    # Подгружаем стерильный датасет V02 (без варнингов в консоли)
    dataloader = get_dataloader_v02()
    
    # Поднимаем трансформер с активированным градиентным чекпоинтингом
    lora_model = FluxLoraCoreV02.init_transformer_with_lora()
    
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TrainConfig.LEARNING_RATE, weight_decay=1e-2)
    
    print(f"[ОБТ] Реактор обкатки V02 запущен на {TrainConfig.RESOLUTION}px. Цель: {TrainConfig.MAX_TRAIN_STEPS} шагов.")
    
    global_step = 0
    epoch = 0
    
    while global_step < TrainConfig.MAX_TRAIN_STEPS:
        epoch += 1
        print(f"[ОБТ] Начало эпохи плавки №{epoch}")
        
        for batch in dataloader:
            latents = batch["latents"].to(device="cuda", dtype=torch.bfloat16)
            prompt_embeds = batch["prompt_embeds"].to(device="cuda", dtype=torch.bfloat16)
            
            b, c, h, w = latents.shape
            noise = torch.randn_like(latents)
            
            # Расчет нелинейного времени и блендинг траектории шума
            t = FluxFlowMathV01.generate_train_timesteps(b, device="cuda")
            noisy_latents, target_flow = FluxFlowMathV01.blend_noise_and_latents(latents, noise, t)
            
            # Упаковка в патчи
            packed_noisy_latents = pack_latents_to_patches(noisy_latents)
            packed_target_flow = pack_latents_to_patches(target_flow)
            
            # Временные атрибуты и координатные сетки патчей с батч-размерностью
            img_ids_raw = generate_flux_img_ids(h, w, device="cuda")
            img_ids = img_ids_raw.unsqueeze(0).repeat(b, 1, 1)
            timesteps_attr = t.squeeze().view(-1) * 1000.0
            
            # СНАЙПЕРСКИЙ ИНТ-ФИКС: Извлекаем raw integer количества токенов (256)
            txt_len = int(prompt_embeds.shape[1])
            txt_ids_raw = torch.zeros(txt_len, 3, device="cuda", dtype=torch.bfloat16)
            txt_ids = txt_ids_raw.unsqueeze(0).repeat(b, 1, 1)
            
            # Генерируем пустой pooled_projections формы [B, 768]
            pooled_projections = torch.zeros(b, 768, device="cuda", dtype=torch.bfloat16)
# === БЛОК 2 ФИНАЛ ===
