import random
import json
import torch
from datetime import datetime
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from logger import chat_logger

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
        # Ограничиваем историю
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_context_aware_response(self, tag, probability_value, intent, user_message):
        response = random.choice(intent['responses'])
        
        # Если пользователь сказал "нет" на уточняющий вопрос
        if user_message.lower() in ['нет', 'не', 'не то', 'неправильно'] and self.previous_tag:
            fallback_responses = [
                "Понял, ошибся. Можете уточнить ваш вопрос?",
                "Извините, не угадал. О чем именно вы хотели спросить?",
                "Попробую еще раз. Сформулируйте, пожалуйста, по-другому.",
                "Не совсем понял. Можете задать вопрос другими словами?"
            ]
            return random.choice(fallback_responses)
        
        # Если низкая вероятность, добавляем уточнение
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

def chat():
    chat_logger.info("=== ЗАПУСК ЧАТ-БОТА ===")
    
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

    chat_logger.info(f"Модель загружена: {len(all_words)} слов, {len(tags)} тегов")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "NewYearBot"
    chat_logger.info("Чат-бот готов к работе")
    
    print(f"{bot_name}: Давайте начнем! (напишите 'выход' чтобы закончить)")
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_logger.info(f"Начата сессия: {session_id}")
    
    context = ChatContext()

    while True:
        sentence = input("Вы: ")
        if sentence.lower() == "выход":
            chat_logger.info(f"Сессия завершена: {session_id}")
            break

        chat_logger.info(f"Вопрос: '{sentence}'")

        # Обработка сообщения
        tokens = tokenize(sentence)
        X = bag_of_words(tokens, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        probability_value = prob.item()
        
        chat_logger.info(f"Предсказание: тег='{tag}', вероятность={probability_value:.4f}")

        # УЛУЧШЕННАЯ ЛОГИКА С КОНТЕКСТОМ
        response = None
        
        if probability_value > 0.85:
            # Высокая уверенность
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = context.get_context_aware_response(tag, probability_value, intent, sentence)
                    break
        
        elif probability_value > 0.65:
            # Средняя уверенность
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = context.get_context_aware_response(tag, probability_value, intent, sentence)
                    break
        
        elif probability_value > 0.45:
            # Низкая уверенность
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = context.get_context_aware_response(tag, probability_value, intent, sentence)
                    break
        
        else:
            # Очень низкая уверенность
            fallback_responses = [
                "Извините, я не совсем понял. Можете переформулировать вопрос о Новом годе?",
                "Не уверен, что правильно понял. Можете уточнить, что вас интересует?",
                "Пожалуйста, задайте вопрос по-другому. Я специализируюсь на новогодней тематике!",
                "Не совсем понял ваш вопрос. Можете перефразировать?",
                "Кажется, я не уловил суть. Можете повторить вопрос другими словами?"
            ]
            response = random.choice(fallback_responses)

        if response:
            print(f"{bot_name}: {response}")
            chat_logger.info(f"Ответ: '{response}' (уверенность: {probability_value:.4f}, тег: {tag})")
            
            # Сохраняем контекст
            context.add_message(sentence, response, tag, probability_value)
            context.previous_tag = tag
        
        print("---")

if __name__ == "__main__":
    chat()