@echo off
chcp 65001 >nul
title Установка пакетов для чатбота

echo ========================================
echo    Установка пакетов для чатбота
echo ========================================

:: Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не установлен или не добавлен в PATH
    pause
    exit /b 1
)

echo.
echo Проверка и установка пакетов...

:: Установка пакетов
echo.
echo [1/6] Установка torch 2.5.1 для CPU...
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2/6] Установка numpy 2.0.1...
pip install numpy==2.0.1

echo.
echo [3/6] Установка nltk 3.9.2...
pip install nltk==3.9.2

echo.
echo [4/6] Установка tqdm 4.67.1...
pip install tqdm==4.67.1

echo.
echo [5/6] Установка python-telegram-bot...
pip install python-telegram-bot

echo.
echo [6/6] Установка моделей NLTK...
python -c "import nltk; nltk.download('punkt', quiet=False); nltk.download('punkt_tab', quiet=False)"

echo.
echo ========================================
echo Установка завершена успешно!
echo ========================================

echo.
echo Проверка установленных пакетов:
python -c "
try:
    import torch, numpy, nltk, tqdm
    print(f'✓ torch: {torch.__version__}')
    print(f'✓ numpy: {numpy.__version__}')
    print(f'✓ nltk: {nltk.__version__}')
    print(f'✓ tqdm: {tqdm.__version__}')
    
    try:
        import telegram
        print('✓ python-telegram-bot: установлен')
    except:
        print('✗ python-telegram-bot: не установлен')
        
except Exception as e:
    print(f'Ошибка при проверке: {e}')
"

pause