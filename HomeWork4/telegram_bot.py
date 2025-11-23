import random
import json
import torch
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from logger import chat_logger

# Контекст
class ChatContext:
    def __init__(self):
        self.previous_tag = None
        self.conversation_history = []
        self.max_history = 5
    
    def add_message(self, user_message, bot_response, tag, probability):
        self.conversation_history.append({
            'user': user_message,
            'bot': bot_response,
            'tag': tag,
            'probability': probability
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_context_aware_response(self, tag, probability_value, intent, user_message):
        response = random.choice(intent['responses'])
        
        if user_message.lower() in ['нет', 'не', 'не то', 'неправильно'] and self.previous_tag:
            fallback_responses = [
                "Понял, ошибся. Можете уточнить ваш вопрос?",
                "Извините, не угадал. О чем именно вы хотели спросить?",
                "Попробую еще раз. Сформулируйте, пожалуйста, по-другому.",
                "Не совсем понял. Можете задать вопрос другими словами?"
            ]
            return random.choice(fallback_responses)
        
        elif probability_value < 0.7:
            clarification_questions = {
                'детский_праздник': "Кажется, вы спрашиваете о развлечениях для детей? ",
                'подарки': "Если я правильно понял, речь о подарках? ",
                'праздничная_кухня': "Вы имеете в виду новогодние рецепты? ",
                'новогодние_фильмы': "Речь идет о фильмах для праздника? ",
                'праздничное_настроение': "Вы спрашиваете о новогоднем настроении? "
            }
            question = clarification_questions.get(tag, "Уточните, пожалуйста, о чем именно вы хотите узнать? ")
            response = question + response
        
        return response

# Класс для управления сессиями пользователей
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
    
    def is_active(self, user_id):
        return self.active_sessions.get(user_id, True)
    
    def stop_session(self, user_id):
        self.active_sessions[user_id] = False
    
    def start_session(self, user_id):
        self.active_sessions[user_id] = True

# Создаем менеджер сессий
session_manager = SessionManager()

# Инициализация модели (выполняется один раз при запуске)
def initialize_model():
    chat_logger.info("=== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ДЛЯ TELEGRAM ===")
    
    with open('intents.json', 'r', encoding='utf-8') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE, weights_only=True)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    chat_logger.info(f"Модель загружена: {len(all_words)} слов, {len(tags)} тегов")
    
    return model, intents, all_words, tags, device

# Глобальные переменные для модели
model, intents, all_words, tags, device = initialize_model()

# Функция для получения списка тем из тегов
def get_topics_from_tags(intents_data):
    """Генерирует список тем для приветственного сообщения на основе тегов из intents.json"""
    # Эмодзи для разных категорий 
    emojis = ['🎄', '🎅', '🎁', '🍪', '🎬', '✨', '🏠', '📜', '💝', '👗', '🎵', '⭐', '🌟', '❄️', '🔥']
    
    topics_list = []
    
    for intent in intents_data['intents']:
        tag = intent['tag']
        # Преобразуем тег в читаемый формат
        readable_topic = tag.replace('_', ' ').title()
        
        # Берем эмодзи по кругу или случайный
        emoji = emojis[len(topics_list) % len(emojis)]
        
        # Формируем строку темы
        topic_line = f"{emoji} {readable_topic}"
        topics_list.append(topic_line)
    
    return topics_list

# Функция для создания приветственного текста
def create_welcome_text():
    """Создает приветственное сообщение с темами из intents.json"""
    topics_list = get_topics_from_tags(intents)
    
    # Формируем текст
    welcome_lines = [
        "🎄 Привет! Я NewYearBot - ваш помощник в подготовке к Новому Году!",
        "",
        "Я могу помочь вам с следующими темами:",
        ""
    ]
    
    # Добавляем темы
    for topic in topics_list:
        welcome_lines.append(topic)
    
    # Добавляем инструкции
    welcome_lines.extend([
        "",
        "Просто задайте ваш вопрос или используйте команды:",
        "/help - показать справку",
        "/stop - приостановить диалог"
    ])
    
    return "\n".join(welcome_lines)

# Обработчик команды /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session_manager.start_session(user_id)
    
    # Создаем приветственное сообщение
    welcome_text = create_welcome_text()
    
    await update.message.reply_text(welcome_text)
    chat_logger.info(f"Пользователь {update.effective_user.id} запустил бота")

# Обработчик команды /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = "\n".join([
        "📖 Доступные команды:",
        "/start - начать общение",
        "/help - показать эту справку", 
        "/stop - приостановить диалог",
        "",
        "Просто напишите ваш вопрос о Новом годе, и я постараюсь помочь!",
        "Мои знания включают все аспекты подготовки к празднику."
    ])
    await update.message.reply_text(help_text)

# Обработчик команды /stop
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session_manager.stop_session(user_id)
    
    stop_messages = [
        "🛑 Диалог приостановлен. Напишите любое сообщение, чтобы продолжить!",
        "⏸️ Остановлено. Когда будете готовы продолжить - просто напишите сообщение!",
        "💤 Перехожу в режим ожидания. Для возобновления отправьте любое сообщение."
    ]
    
    await update.message.reply_text(random.choice(stop_messages))
    chat_logger.info(f"Пользователь {user_id} приостановил диалог")

# Основной обработчик сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text
    
    # Проверяем, не остановлен ли диалог
    if not session_manager.is_active(user_id):
        session_manager.start_session(user_id)  # Возобновляем диалог
        await update.message.reply_text("✅ Диалог возобновлен! Задавайте ваш вопрос о Новом годе!")
        return
    
    chat_logger.info(f"Сообщение от {user_id}: '{user_message}'")
    
    # Инициализируем контекст для пользователя
    if 'chat_context' not in context.user_data:
        context.user_data['chat_context'] = ChatContext()
    
    chat_context = context.user_data['chat_context']
    
    # Обработка сообщения моделью
    tokens = tokenize(user_message)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    probability_value = prob.item()
    
    chat_logger.info(f"Предсказание для {user_id}: тег='{tag}', вероятность={probability_value:.4f}")

    # Логика выбора ответа
    response = None
    
    if probability_value > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = chat_context.get_context_aware_response(tag, probability_value, intent, user_message)
                break
    
    elif probability_value > 0.65:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = chat_context.get_context_aware_response(tag, probability_value, intent, user_message)
                break
    
    elif probability_value > 0.45:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = chat_context.get_context_aware_response(tag, probability_value, intent, user_message)
                break
    
    else:
        fallback_responses = [
            "Извините, я не совсем понял. Можете переформулировать вопрос о Новом годе?",
            "Не уверен, что правильно понял. Можете уточнить, что вас интересует?",
            "Пожалуйста, задайте вопрос по-другому. Я специализируюсь на новогодней тематике!",
            "Не совсем понял ваш вопрос. Можете перефразировать?",
            "Кажется, я не уловил суть. Можете повторить вопрос другими словами?"
        ]
        response = random.choice(fallback_responses)

    if response:
        await update.message.reply_text(response)
        chat_logger.info(f"Ответ пользователю {user_id}: '{response}' (уверенность: {probability_value:.4f}, тег: {tag})")
        
        # Сохраняем контекст
        chat_context.add_message(user_message, response, tag, probability_value)
        chat_context.previous_tag = tag

# Обработчик ошибок
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_logger.error(f"Ошибка при обработке сообщения: {context.error}")
    await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")

# Основная функция
def main():
    # токен
    BOT_TOKEN = "в целях  безопасности удален"
    
    # Создаем приложение
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    chat_logger.info("Telegram бот запущен...")
    print("Бот запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()