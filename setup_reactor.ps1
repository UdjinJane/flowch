# =====================================================================
# 🔱 БОРТОВОЙ АВТОМАТ СНАБЖЕНИЯ ЯДРА CHROMA1: setup_reactor.ps1
# ОРИЕНТИРОВАН СТРОГО НА WINDOWS 10 / WINDOWS SERVER (64-bit AMD64)
# =====================================================================

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=== [ОТК] СТАРТ ТЕХНОЛОГИЧЕСКОЙ САНАЦИИ РЕАКТОРА ===" -ForegroundColor Cyan

# ---------------------------------------------------------------------
# ШАГ 1: АНАЛИЗ ГЛОБАЛЬНОЙ СРЕДЫ И ФИЛЬТРАЦИЯ ДИВЕРСИЙ PYTHON
# ---------------------------------------------------------------------
Write-Host "[1/5] Проверка глобального интерпретатора..." -ForegroundColor Yellow
$GlobalPythonVer = & python --version 2>$null

if ($GlobalPythonVer -match "3\.13") {
    Write-Host "🚨 ВНИМАНИЕ: Обнаружена диверсионная ветка Python 3.13!" -ForegroundColor Red
    Write-Host "Автоматический запуск из дефолтных путей запрещен во избежание краша импортов." -ForegroundColor Yellow
}

# Снайперский поиск эталонного 64-битного ядра Python 3.10 в трюмах системы
$TargetPython = "C:\Users\Udjin\AppData\Local\Programs\Python\Python310\python.exe"

if (-not (Test-Path $TargetPython)) {
    Write-Host "❌ КРИТИЧЕСКАЯ ОШИБКА: Эталонный путь Python 3.10 не найден!" -ForegroundColor Red
    Write-Host "Маршрут $TargetPython пуст. Проверьте менеджер версий." -ForegroundColor Yellow
    Exit
}

$RealArch = & $TargetPython -c "import platform; print(platform.architecture()[0])"
Write-Host "[УСПЕХ] Найдено стабильное ядро: Python 3.10 ($RealArch)" -ForegroundColor Green

# ---------------------------------------------------------------------
# ШАГ 2: АННИГИЛЯЦИЯ СТАРОГО ТРЮМА И СБОРКА ЧИСТОГО КОНТУРА VENV
# ---------------------------------------------------------------------
Write-Host "[2/5] Подготовка изолированной кабельной трассы VENV..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Write-Host "Обнаружены остатки старой среды. Производится полная зачистка..." -ForegroundColor Black
    Remove-Item -Path ".\venv" -Recurse -Force
}

Write-Host "Разворачивание стерильной 64-битной venv..." -ForegroundColor Black
& $TargetPython -m venv venv

# Перепроверяем, что рубка управления Scripts создалась корректно
$LocalPython = ".\venv\Scripts\python.exe"
$LocalPip    = ".\venv\Scripts\pip.exe"

if (-not (Test-Path $LocalPython)) {
    Write-Host "❌ КРИТИЧЕСКАЯ ОШИБКА: Сборка venv сорвана, бинарники Scripts отсутствуют!" -ForegroundColor Red
    Exit
}

# ---------------------------------------------------------------------
# ШАГ 3: ОБНОВЛЕНИЕ БАЗОВЫХ СЕРВОПРИВОДОВ PIP
# ---------------------------------------------------------------------
Write-Host "[3/5] Форсирование внутренних утилит pip..." -ForegroundColor Yellow
& $LocalPython -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------
# ШАГ 4: ЗАЛИВКА ТЯЖЕЛОГО ТЕНЗОРНОГО ЯДРА (ВОЛАПЮК-МАГИСТРАЛЬ)
# ---------------------------------------------------------------------
Write-Host "[4/5] Подключение шифрованной шины снабжения PyTorch..." -ForegroundColor Yellow
Write-Host "Полосатые купальники начинают перетаскивать 2.4 ГБ CUDA-топлива..." -ForegroundColor Black

# ДЕШИФРАТОР ДЛЯ АЛЬТЕРОВ: https://pytorch.org
$VolapukIndex = "https://дувнлоад-тчк-пыторч-тчк-орг/whl/cu121"
$CleanIndex   = $VolapukIndex.Replace("дувнлоад-тчк-пыторч-тчк-орг", "download.pytorch.org").Replace("-тчк-", ".")

& $LocalPip install torch==2.5.1 torchvision==0.20.1 --index-url $CleanIndex

# ---------------------------------------------------------------------
# ШАГ 5: НАКАТ РАБОЧЕЙ ОБВЯЗКИ И ДАТЧИКОВ НЕПОПРАВИМЫХ УЛУЧШЕНИЙ
# ---------------------------------------------------------------------
Write-Host "[5/5] Накат маршевой обвязки и утилит квантования..." -ForegroundColor Yellow

# Фиксируем совместимый torchao=0.7.0 во избежание конфликтов мантиссы и ошибки int1
& $LocalPip install diffusers==0.39.0 peft safetensors pillow bitsandbytes accelerate transformers psutil
& $LocalPip install torchao==0.7.0 --no-deps

# ---------------------------------------------------------------------
# ФИНАЛЬНАЯ ВЕРИФИКАЦИЯ И ХОЛОСТНОЙ ПОДЖИГ
# ---------------------------------------------------------------------
Write-Host "`n=== ТЕЛЕМЕТРИЯ ГОТОВНОСТИ РЕАКТОРА ===" -ForegroundColor Cyan
$CudaStatus = & $LocalPython -c "import torch; print(torch.cuda.is_available())"

if ($CudaStatus -match "True") {
    $GpuName = & $LocalPython -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "[УСПЕХ] Контур загерметизирован! CUDA Активна." -ForegroundColor Green
    Write-Host "[УСПЕХ] Обнаружен маршевый движок: $GpuName" -ForegroundColor Green
    Write-Host "Реактор готов к пуску. Инструкция Василис полностью исполнена.`n" -ForegroundColor Cyan
} else {
    Write-Host "🚨 АВАРИЯ: Тензорное ядро встало в режим CPU! Проверьте драйверы NVIDIA." -ForegroundColor Red
}
