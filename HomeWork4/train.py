import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from logger import train_logger  # Импортируем логгер

def train_model():
    train_logger.info("=== НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ ===")
    
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    
    # Логируем загрузку данных
    train_logger.info("Загрузка данных из intents.json")
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Приведение регистров
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Логируем статистику данных
    train_logger.info(f"Данные загружены: {len(xy)} паттернов, {len(tags)} тегов, {len(all_words)} уникальных слов")
    train_logger.info(f"Теги: {tags}")
    train_logger.info(f"Пример слов: {all_words[:10]}")

    # create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Hyper-parameters 
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 128
    output_size = len(tags)
    
    # Логируем параметры модели
    train_logger.info(f"Параметры модели: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
    train_logger.info(f"Параметры обучения: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_logger.info(f"Используемое устройство: {device}")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_logger.info("Начало обучения...")
    
    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            avg_loss = epoch_loss / len(train_loader)
            train_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    final_loss = epoch_loss / len(train_loader)
    train_logger.info(f"Обучение завершено. Финальный loss: {final_loss:.4f}")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    train_logger.info(f"Модель сохранена в файл: {FILE}")
    train_logger.info("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===\n")

if __name__ == "__main__":
    train_model()