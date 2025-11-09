import torch
import torch.nn as nn
import torch.optim as optim
import dataset.dataset_creation as dc
from torch.utils.data import DataLoader, Subset
from unet.unet_model import UNet
import time
import numpy as np
import print.pr as pr

'''
# 1. Уменьшаем датасет (берем только маленькую часть)
dataset = dc.BrainDataset(root_dir='./examples')
# Берем только первые 20 примеров для быстрого теста
indices = torch.arange(20)  # всего 20 примеров
small_dataset = Subset(dataset, indices)

# 2. Уменьшаем batch size и убираем shuffle для стабильности
dataloader = DataLoader(small_dataset, batch_size=2, shuffle=False)
'''
#1. Формируем датасет
dataset = dc.BrainDataset(root_dir='./examples')

#2. Подгружаем данные
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
pr.log_print(f"Размер датасета: {len(dataset)} примеров")

# 3. Создаем упрощенную модель (меньше каналов)
model = UNet(n_channels=1, n_classes=1)
pr.log_print(f"Модель: UNet с {sum(p.numel() for p in model.parameters())} параметрами")

# 4. Функция потерь и оптимизатор, введем гиперпараметры 
pos_weight = torch.tensor([10.0]).  # балансировка классов
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # повышаем штраф за ошибку
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)  # регулируем скорость обучения + регуляризация

pr.log_print(f"Используем взвешенный loss с pos_weight={pos_weight.item()}")
pr.log_print(f"И регуляризацию с weight_decay=1e-5")

# 5. Параметры обучения
num_epochs = 100          # количество эпох
patience = 10             # сколько эпох ждать ухудшения перед остановкой
min_delta = 0.001         # минимальное улучшение, которое считаем значимым
min_good_loss = 0.08      # если loss ниже этого значения - останавлива

# Переменные для ранней остановки
best_loss = float('inf')
epochs_no_improve = 0

pr.log_print("Начинаем обучение...")
pr.log_print(f"Параметры: максимум {num_epochs} эпох, patience={patience}, min_good_loss={min_good_loss}, batch_size={dataloader.batch_size}, dataset_size={len(dataset)}")
pr.log_print(f"Ранняя остановка сработает если: loss <= {min_good_loss}, или нет улучшений {patience} эпох, или достигнут {num_epochs} эпох")
pr.log_print("-" * 60)

# проверка данных
pr.log_print("ПРОВЕРКА ДАННЫХ по нескольким батчам:")
for i, (images, masks) in enumerate(dataloader):
    pr.log_print(f"Батч {i+1}:")
    pr.log_print(f"  Изображения: [{images.min():.3f}, {images.max():.3f}]")
    pr.log_print(f"  Маски: [{masks.min():.3f}, {masks.max():.3f}]")
    
    # Считаем процент ненулевых масок
    non_zero = (masks > 0).sum().item()
    total_pixels = masks.numel()
    pr.log_print(f"  Пикселей с опухолью: {non_zero}/{total_pixels} ({non_zero/total_pixels*100:.2f}%)")
    
    if i == 2:  # Проверим 3 батча
        break

# Засекаем время начала обучения
start_time = time.time()
# Логирование начала обучения
pr.log_print("=" * 70)
pr.log_print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ UNet")
pr.log_print("=" * 70)
pr.log_print(f"Время начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")
pr.log_print(f"Датасет: ./examples")

# Цикл обучения
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
       
    for batch_idx, (images, masks) in enumerate(dataloader):
        # важно: нормализация изображений
        images = images / 255.0           # [0, 255] → [0, 1]
        masks = masks / 255.0             # [0, 255] → [0, 1]
        masks = (masks > 0.5).float()     # бинаризация: >128 = 1, иначе 0

        # переносим данные НА GPU
        images = images.to(device)
        masks = masks.to(device)
       
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        pr.log_print(f'Эпоха {epoch+1}/{num_epochs} | Батч {batch_idx+1}/{len(dataloader)} | Loss: {batch_loss:.4f}')
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = epoch_loss / len(dataloader)
    
    pr.log_print(f"Эпоха {epoch+1} завершена!")
    pr.log_print(f"Средний Loss: {avg_loss:.4f}")
    pr.log_print(f"Время эпохи: {epoch_time:.1f} сек")
    
    # Проверка улучшения loss
    if avg_loss < best_loss - min_delta:
        pr.log_print(f"УЛУЧШЕНИЕ! Loss уменьшился с {best_loss:.4f} до {avg_loss:.4f}")
        best_loss = avg_loss
        epochs_no_improve = 0

        # Сохраняем модель
        torch.save(model.state_dict(), 'unet_brain_best.pth')

         # Сохраняем метаданные о лучшей модели
        best_model_info = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'best_loss': best_loss,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_size': len(dataset),
            'batch_size': dataloader.batch_size,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'hyperparameters': {
                'pos_weight': pos_weight.item(),
                'learning_rate': 1e-5,
                'weight_decay': 1e-5
            }
        }
        # Записываем информацию о лучшей модели в отдельный файл
        with open('best_model_info.txt', 'w', encoding='utf-8') as f:
            f.write("ИНФОРМАЦИЯ О ЛУЧШЕЙ МОДЕЛИ:\n")
            f.write("=" * 50 + "\n")
            for key, value in best_model_info.items():
                if key == 'hyperparameters':
                    f.write("Гиперпараметры:\n")
                    for hp_key, hp_value in value.items():
                        f.write(f"  {hp_key}: {hp_value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        pr.log_print("Сохранена лучшая модель 'unet_brain_best.pth'")
        pr.log_print(f"Информация о лучшей модели сохранена в 'best_model_info.txt'")

    else:
        epochs_no_improve += 1
        pr.log_print(f"Ухудшение №{epochs_no_improve}. Лучший loss: {best_loss:.4f}, текущий: {avg_loss:.4f}")

    # проверка ранней оставновки (три условия)
    stop_training = False
    stop_reason = ""

    # Условие 1: Достигли целевого качества
    if avg_loss <= min_good_loss:
        stop_training = True
        stop_reason = f"Достигнут хороший loss: {avg_loss:.4f} <= {min_good_loss}"

    # Условие 2: Застряли на одном уровне  
    elif epochs_no_improve >= patience:
        stop_training = True
        stop_reason = f"Loss не улучшался {patience} эпох подряд"

    # Условие 3: Слишком долго учимся
    elif epoch + 1 >= num_epochs:
        stop_training = True
        stop_reason = f"Достигнут лимит в {num_epochs} эпох"

    if stop_training:
        pr.log_print(f"РАННЯЯ ОСТАНОВКА! {stop_reason}")
        pr.log_print(f"Обучение остановлено на эпохе {epoch+1}")
        break
    
    pr.log_print("-" * 60)

# Общее время обучения
total_time = time.time() - start_time

pr.log_print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
pr.log_print(f"Итоговые результаты:")
pr.log_print(f"   Пройдено эпох: {epoch+1}")
pr.log_print(f"   Лучший Loss: {best_loss:.4f}")
pr.log_print(f"   Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
pr.log_print(f"   Размер датасета: {len(dataset)} примеров")
pr.log_print(f"   Batch size: {dataloader.batch_size}")
pr.log_print(f"   Общее время обучения: {total_time:.1f} сек ({total_time/60:.1f} мин)")
pr.log_print(f"   Среднее время на эпоху: {total_time/(epoch+1):.1f} сек" if epoch > 0 else "   Среднее время на эпоху: N/A")

# Сохраняем финальную модель
torch.save(model.state_dict(), 'unet_brain_final.pth')
pr.log_print("Модель сохранена как 'unet_brain_final.pth'")
pr.log_print("Лучшая модель сохранена как 'unet_brain_best.pth'")

pr.close()