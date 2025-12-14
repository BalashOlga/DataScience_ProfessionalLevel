# bot.py
import os
import logging
from pathlib import Path
from io import BytesIO
import cv2
import numpy as np
import json  # Добавили импорт json

import torch
from PIL import Image
import torchvision.transforms as transforms
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import ReplyKeyboardMarkup, KeyboardButton

# Используем улучшенную архитектуру и ухудшение качества
from model_architecture_improved import ImageEnhancerImproved
from lquality import degrade_single_image

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot_logs.log'
)
logger = logging.getLogger(__name__)

# Относительные пути
MODEL_PATH = "training_results_improved/models/best_model.pth"
CONFIG_PATH = "training_results_improved/config.json"  # Добавили путь к конфигурации

# Клавиатура меню
menu_keyboard = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🔄 Обработать новое фото")],
        [KeyboardButton("ℹ️ Помощь"), KeyboardButton("🚫 Выход")]
    ],
    resize_keyboard=True
)

def load_config():
    """Загружает конфигурацию из config.json"""
    config_path = Path(CONFIG_PATH)
    
    if not config_path.exists():
        logger.warning(f"Файл конфигурации не найден: {CONFIG_PATH}, использую значения по умолчанию")
        return {
            'NUM_RESIDUAL_BLOCKS': 8,
            'NUM_FEATURES': 64,
            'DROPOUT_RATE': 0.1
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Конфигурация загружена из {CONFIG_PATH}")
            return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        return {
            'NUM_RESIDUAL_BLOCKS': 8,
            'NUM_FEATURES': 64,
            'DROPOUT_RATE': 0.1
        }

# Загрузка модели
def load_best_model():
    """Загружает лучшую модель"""
    model_path = Path(MODEL_PATH)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # СНАЧАЛА пробуем загрузить конфигурацию
    config = load_config()
    
    # Получаем параметры модели из конфигурации
    num_blocks = config.get('NUM_RESIDUAL_BLOCKS', 8)
    num_features = config.get('NUM_FEATURES', 64)
    dropout_rate = config.get('DROPOUT_RATE', 0.1)
    
    logger.info(f"Параметры модели из config.json: blocks={num_blocks}, features={num_features}, dropout={dropout_rate}")
    
    # Создаем модель с параметрами из конфигурации
    model = ImageEnhancerImproved(
        num_residual_blocks=num_blocks,
        num_features=num_features,
        dropout_rate=dropout_rate
    )
    
    # Загружаем веса модели
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Загружаем веса
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Модель успешно загружена")
    return model, device

# Инициализация модели
try:
    model, device = load_best_model()
    logger.info("Модель инициализирована успешно")
except Exception as e:
    logger.error(f"Не удалось загрузить модель: {e}")
    print(f"ОШИБКА: {e}")
    exit(1)

# Загрузка классификатора для обнаружения лиц (OpenCV)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Загружен классификатор лиц OpenCV")
except:
    face_cascade = None
    logger.warning("Не удалось загрузить классификатор лиц OpenCV")

def detect_and_crop_face(image_pil):
    """
    Обнаруживает лицо на изображении и вырезает его в прямоугольник 178x218
    """
    # Конвертируем PIL в OpenCV формат
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Обнаруживаем лица
    faces = []
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
    
    if len(faces) > 0:
        # Берем самое большое лицо
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Добавляем немного отступов вокруг лица (20%)
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(image_cv.shape[1] - x, w + 2 * padding_x)
        h = min(image_cv.shape[0] - y, h + 2 * padding_y)
        
        # Обрезаем лицо с отступами
        face_cropped = image_cv[y:y+h, x:x+w]
        
        # Конвертируем обратно в PIL
        face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        
        # Масштабируем до 178x218 с искажением (для модели)
        face_resized = face_pil.resize((178, 218), Image.Resampling.LANCZOS)
        
        logger.info(f"Найдено лицо: {len(faces)} штук, вырезано: {w}x{h}")
        return face_resized, True, (x, y, w, h)
    
    # Если лицо не найдено
    logger.info("Лицо не найдено")
    return None, False, None

def prepare_face_image(image_pil):
    """
    Подготавливает изображение лица для модели:
    1. Обнаруживает и вырезает лицо 178x218
    2. Если лицо найдено - ухудшает его качество (как в обучении)
    3. Преобразует в тензор
    """
    # Обнаруживаем и вырезаем лицо
    face_img, face_found, face_coords = detect_and_crop_face(image_pil)
    
    if not face_found:
        # Если лицо не найдено, возвращаем сообщение об ошибке
        return {
            'face': None,
            'face_tensor': None,
            'input_tensor': None,
            'face_found': False,
            'error_message': "❌ Лицо не найдено на фотографии. Пожалуйста, отправьте фото с четко видимым лицом."
        }
    
    # Ухудшаем качество лица (как в обучении)
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    degraded_cv = degrade_single_image(face_cv)
    
    # Конвертируем обратно в PIL
    degraded_pil = Image.fromarray(cv2.cvtColor(degraded_cv, cv2.COLOR_BGR2RGB))
    
    # Преобразуем в тензор
    to_tensor = transforms.ToTensor()
    degraded_tensor = to_tensor(degraded_pil)
    
    # Добавляем batch dimension
    input_tensor = degraded_tensor.unsqueeze(0).to(device)
    
    return {
        'face': degraded_pil,  # Ухудшенное лицо
        'face_original': face_img,  # Оригинальное лицо (без ухудшения)
        'face_tensor': degraded_tensor,
        'input_tensor': input_tensor,
        'face_found': face_found,
        'face_coords': face_coords
    }

def enhance_image(input_tensor):
    """Улучшает изображение с помощью модели"""
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
    
    enhanced_tensor = enhanced_tensor.squeeze(0).clamp(0, 1)
    return enhanced_tensor

def tensor_to_image(tensor):
    """Конвертирует тензор в PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    to_pil = transforms.ToPILImage()
    return to_pil(tensor.cpu())

def create_face_comparison(original_face, enhanced_face):
    """
    Создает сравнение лица до и после
    """
    # Убеждаемся, что оба изображения одного размера
    if original_face.size != enhanced_face.size:
        enhanced_face = enhanced_face.resize(original_face.size, Image.Resampling.LANCZOS)
    
    # Создаем коллаж (ухудшенное лицо | улучшенное лицо)
    total_width = original_face.width * 2 + 20
    height = original_face.height
    
    # Создаем фон
    collage = Image.new('RGB', (total_width, height), (50, 50, 50))
    
    # Вставляем оригинальное лицо
    collage.paste(original_face, (5, 0))
    
    # Вставляем улучшенное лицо
    collage.paste(enhanced_face, (original_face.width + 15, 0))
    
    # Добавляем подписи
    from PIL import ImageDraw, ImageFont
    
    try:
        draw = ImageDraw.Draw(collage)
        font = ImageFont.load_default()
        
        # Подпись для оригинала
        draw.text((10, 10), "ДО", fill=(255, 100, 100), font=font)
        
        # Подпись для улучшенного
        draw.text((original_face.width + 20, 10), "ПОСЛЕ", fill=(100, 255, 100), font=font)
        
        # Размер внизу
        size_text = f"{original_face.width}×{original_face.height}"
        text_width = draw.textlength(size_text, font=font)
        draw.text(
            (total_width - text_width - 10, height - 20), 
            size_text, 
            fill=(200, 200, 200), 
            font=font
        )
    except:
        pass
    
    return collage

# Обработчики команд бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    logger.info(f"Пользователь {user.id} начал работу с ботом")
    
    # Загружаем конфигурацию для отображения параметров
    config = load_config()
    num_blocks = config.get('NUM_RESIDUAL_BLOCKS', 8)
    num_features = config.get('NUM_FEATURES', 64)
    dropout_rate = config.get('DROPOUT_RATE', 0.1)
    
    welcome_text = (
        "👋 Привет! Я бот для улучшения качества лиц на фотографиях.\n\n"
        "✨ **Параметры модели:**\n"
        f"• Residual блоков: {num_blocks}\n"
        f"• Features: {num_features}\n"
        f"• Dropout: {dropout_rate}\n\n"
        "Как это работает:\n"
        "1. Вы отправляете фотографию с лицом\n"
        "2. Я нахожу лицо и вырезаю его (178×218)\n"
        "3. Улучшаю качество нейросетью\n"
        "4. Показываю сравнение лица до и после\n\n"
        "Отправьте фото с лицом чтобы начать!"
    )
    
    await update.message.reply_text(welcome_text, reply_markup=menu_keyboard)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = (
        "🤖 Бот для улучшения качества лиц\n\n"
        "📸 Отправьте фото с лицом (лучше портретное)\n"
        "👤 Я найду лицо и вырежу его в размер 178×218\n"
        "✨ Нейросеть улучшит качество этого лица\n"
        "🔄 Покажу сравнение до и после\n\n"
        "⚠️ Важно: фото должно содержать четко видимое лицо\n"
        "Размер для обработки фиксированный: 178×218 пикселей\n\n"
        "Используйте кнопки меню для навигации."
    )
    
    await update.message.reply_text(help_text, reply_markup=menu_keyboard)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений (кнопок меню)"""
    text = update.message.text
    user = update.effective_user
    
    if text == "🔄 Обработать новое фото":
        await update.message.reply_text(
            "📸 Отправьте новую фотографию с лицом для обработки.",
            reply_markup=menu_keyboard
        )
        logger.info(f"Пользователь {user.id} запросил новую обработку")
    
    elif text == "ℹ️ Помощь":
        await help_command(update, context)
    
    elif text == "🚫 Выход":
        await update.message.reply_text(
            "👋 До свидания! Чтобы начать заново, отправьте /start",
            reply_markup=None
        )
        logger.info(f"Пользователь {user.id} завершил работу")
    
    else:
        await update.message.reply_text(
            "Используйте кнопки меню или отправьте фото с лицом для обработки.",
            reply_markup=menu_keyboard
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик фотографий"""
    user = update.effective_user
    logger.info(f"Пользователь {user.id} отправил фото")
    
    processing_msg = None
    
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await update.message.reply_text(
            "🔍 Ищу лицо на фото...",
            reply_markup=menu_keyboard
        )
        
        # Получаем фото (берем самое большое)
        photo_file = await update.message.photo[-1].get_file()
        
        # Скачиваем фото
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Открываем изображение
        original_image = Image.open(BytesIO(photo_bytes)).convert('RGB')
        
        # Подготавливаем изображение лица
        prepared = prepare_face_image(original_image)
        
        # Проверяем, найдено ли лицо
        if not prepared['face_found']:
            if processing_msg:
                try:
                    await processing_msg.delete()
                except:
                    pass
            await update.message.reply_text(
                prepared['error_message'],
                reply_markup=menu_keyboard
            )
            logger.info(f"Лицо не найдено для пользователя {user.id}")
            return
        
        # Если лицо найдено - обновляем сообщение
        try:
            if processing_msg:
                await processing_msg.edit_text("✨ Улучшаю качество нейросетью...")
            else:
                await update.message.reply_text("✨ Улучшаю качество нейросетью...", reply_markup=menu_keyboard)
        except Exception as edit_error:
            logger.warning(f"Не удалось отредактировать сообщение: {edit_error}")
            if processing_msg:
                try:
                    await processing_msg.delete()
                except:
                    pass
            processing_msg = await update.message.reply_text(
                "✨ Улучшаю качество нейросетью...",
                reply_markup=menu_keyboard
            )
        
        # Улучшаем изображение лица
        enhanced_tensor = enhance_image(prepared['input_tensor'])
        
        # Конвертируем в PIL Image
        degraded_face_pil = prepared['face']
        enhanced_face_pil = tensor_to_image(enhanced_tensor)
        
        # Создаем коллаж для сравнения
        collage = create_face_comparison(degraded_face_pil, enhanced_face_pil)
        
        # Сохраняем временные файлы
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        collage_path = os.path.join(temp_dir, f"face_comparison_{user.id}.jpg")
        collage.save(collage_path, "JPEG", quality=95)
        
        # Удаляем сообщение "Обрабатываю"
        if processing_msg:
            try:
                await processing_msg.delete()
            except:
                pass
        
        # Готовим текст результата
        result_text = (
            "✅ Лицо найдено и обработано!\n\n"
            "📐 Размер: 178×218 пикселей\n"
            "✨ Качество улучшено нейросетью\n\n"
            "Слева: оригинальное лицо (уменьшенное)\n"
            "Справа: улучшенная версия"
        )
        
        await update.message.reply_text(result_text, reply_markup=menu_keyboard)
        
        with open(collage_path, 'rb') as photo:
            await update.message.reply_photo(
                photo=photo,
                caption="Сравнение качества лица",
                reply_markup=menu_keyboard
            )
        
        # Очищаем временный файл
        if os.path.exists(collage_path):
            os.remove(collage_path)
        
        logger.info(f"Лицо успешно обработано для пользователя {user.id}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке фото для пользователя {user.id}: {e}", exc_info=True)
        
        # Пытаемся удалить сообщение о процессе, если оно есть
        if processing_msg:
            try:
                await processing_msg.delete()
            except:
                pass
        
        error_msg = f"Произошла ошибка: {str(e)}"
        await update.message.reply_text(
            error_msg,
            reply_markup=menu_keyboard
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка в обновлении {update}: {context.error}", exc_info=True)
    
    if update.effective_message:
        await update.effective_message.reply_text(
            "Произошла ошибка. Попробуйте отправить другое фото.",
            reply_markup=menu_keyboard
        )

def main():
    """Запуск бота"""
    # Получаем токен из переменной среды
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("Токен не найден. Установите переменную среды TELEGRAM_BOT_TOKEN")
        print("ОШИБКА: Токен не найден. Установите переменную среды TELEGRAM_BOT_TOKEN")
        print("Пример: set TELEGRAM_BOT_TOKEN=ваш_токен_бота")
        return
    
    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Регистрируем обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("Бот запущен")
    print("🤖 Бот для улучшения качества лиц запущен!")
    print(f"✨ Использует улучшенную модель (параметры из {CONFIG_PATH})")
    print("👤 Специализируется на обработке лиц 178×218")
    print("📸 Отправьте боту /start для начала работы")
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
        print("\n👋 Бот остановлен.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()