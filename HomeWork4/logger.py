import logging
import os
from datetime import datetime

def setup_logging():
    """Настройка системы логирования"""
    
    # Создаем папку для логов если ее нет
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Текущая дата для имени файла
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Логгер для обучения
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)
    
    # Логгер для чата
    chat_logger = logging.getLogger('chat')
    chat_logger.setLevel(logging.INFO)
    
    # Файловые обработчики
    train_handler = logging.FileHandler(f'logs/training_{current_date}.log', encoding='utf-8')
    chat_handler = logging.FileHandler(f'logs/chat_{current_date}.log', encoding='utf-8')
    
    # Установка форматтера
    train_handler.setFormatter(formatter)
    chat_handler.setFormatter(formatter)
    
    # Добавление обработчиков
    train_logger.addHandler(train_handler)
    chat_logger.addHandler(chat_handler)
    
    # Также логируем в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    train_logger.addHandler(console_handler)
    chat_logger.addHandler(console_handler)
    
    return train_logger, chat_logger

# Инициализация логгеров
train_logger, chat_logger = setup_logging()