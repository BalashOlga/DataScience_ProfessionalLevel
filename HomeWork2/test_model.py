import torch
import matplotlib.pyplot as plt
from unet.unet_model import UNet
import dataset.dataset_creation as dc
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

print("=" * 50)
print("ДИАГНОСТИКА МОДЕЛИ")
print("=" * 50)

# 1. Загружаем обученную модель
print("Загружаем модель...")
model = UNet(n_channels=1, n_classes=1)

try:
    checkpoint = torch.load('./models/unet_brain_best.pth', map_location='cpu')
    print(f"Файл модели загружен, размер: {len(checkpoint)} параметров")
    model.load_state_dict(checkpoint)
    print("Веса модели загружены")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

model.eval()
print("Модель в eval режиме")

# 2. Проверка весов модели
print("Проверка весов модели:")
for name, param in model.named_parameters():
    if 'inc' in name and 'weight' in name:
        print(f"Первый слой - {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
        break

# 3. Загружаем данные
print("Загружаем данные...")
dataset = dc.BrainDataset(root_dir='./examples')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. Берем один батч для тестирования
test_images, test_masks = next(iter(dataloader))

# 5. Диагностика данных ДО препроцессинга
print("Данные ДО нормализации:")
print(f"  Изображения: [{test_images.min():.1f}, {test_images.max():.1f}]")
print(f"  Маски: [{test_masks.min():.1f}, {test_masks.max():.1f}]")

# 6. Препроцессинг
test_images = test_images / 255.0
test_masks = test_masks / 255.0
test_masks = (test_masks > 0.5).float()

print("Данные ПОСЛЕ нормализации:")
print(f"  Изображения: [{test_images.min():.3f}, {test_images.max():.3f}]")
print(f"  Маски: [{test_masks.min():.3f}, {test_masks.max():.3f}]")

# 7. Проверка разнообразия батча
print("Проверка разнообразия батча:")
for i in range(2):
    img_mean = test_images[i].mean().item()
    mask_mean = test_masks[i].mean().item()
    print(f"  Изображение {i}: mean={img_mean:.3f}, маска: {mask_mean:.3f}")

# 8. Делаем предсказания
print("Делаем предсказания...")
with torch.no_grad():
    outputs = model(test_images)
    predictions = torch.sigmoid(outputs)

# 9. Детальная статистика предсказаний
print("СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
print(f"  Диапазон: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"  Среднее: {predictions.mean():.6f}")
print(f"  Стандартное отклонение: {predictions.std():.6f}")

# Анализ распределения предсказаний
pred_np = predictions.numpy().flatten()
low_conf = np.sum(pred_np < 0.3) / len(pred_np) * 100
high_conf = np.sum(pred_np > 0.7) / len(pred_np) * 100
mid_conf = np.sum((pred_np >= 0.3) & (pred_np <= 0.7)) / len(pred_np) * 100

print(f"  Предсказания <0.3: {low_conf:.1f}%")
print(f"  Предсказания 0.3-0.7: {mid_conf:.1f}%")
print(f"  Предсказания >0.7: {high_conf:.1f}%")

# 10. Визуализация
print("Создаем визуализацию...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i in range(2):
    # Исходное изображение
    axes[i, 0].imshow(test_images[i][0], cmap='gray')
    axes[i, 0].set_title(f'Исходное (mean: {test_images[i][0].mean():.3f})')
    axes[i, 0].axis('off')
    
    # Реальная маска
    axes[i, 1].imshow(test_masks[i][0], cmap='gray')
    axes[i, 1].set_title(f'Маска (mean: {test_masks[i][0].mean():.3f})')
    axes[i, 1].axis('off')
    
    # Предсказание
    pred_img = axes[i, 2].imshow(predictions[i][0], cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_title(f'Предсказание (mean: {predictions[i][0].mean():.3f})')
    axes[i, 2].axis('off')
    
    # Гистограмма предсказаний
    axes[i, 3].hist(predictions[i][0].flatten().numpy(), bins=50, alpha=0.7, range=(0, 1))
    axes[i, 3].set_title('Распределение предсказаний')
    axes[i, 3].set_xlabel('Значение')
    axes[i, 3].set_ylabel('Частота')

plt.tight_layout()
plt.savefig('model_predictions_diagnostic.png', dpi=150, bbox_inches='tight')
plt.show()

print("Готово! Диагностика сохранена как 'model_predictions_diagnostic.png'")

# 11. Тест на простом случае
print("\n" + "=" * 50)
print("ТЕСТ НА ПРОСТОМ СЛУЧАЕ")
print("=" * 50)

# Создаем простой тест - черное изображение с белым квадратом
simple_image = torch.zeros(1, 1, 256, 256)
simple_image[0, 0, 100:150, 100:150] = 1.0  # белый квадрат

with torch.no_grad():
    simple_output = model(simple_image)
    simple_pred = torch.sigmoid(simple_output)

print(f"Простой тест - предсказание: [{simple_pred.min():.6f}, {simple_pred.max():.6f}]")
print(f"Среднее: {simple_pred.mean():.6f}")

if simple_pred.mean() > 0.1:
    print("Модель реагирует на простые формы")
else:
    print("Модель НЕ реагирует на простые формы")