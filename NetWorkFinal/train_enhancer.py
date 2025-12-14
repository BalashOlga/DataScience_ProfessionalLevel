import torch
import torch.nn as nn
import torch.nn.functional as F  # Добавьте эту строку!
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import cv2
from PIL import Image
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== МОДЕЛЬ =====================
class ResidualBlock(nn.Module):
    """Остаточный блок для глубокого обучения"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class ImageEnhancer(nn.Module):
    """Основная модель для улучшения качества изображений"""
    def __init__(self, num_residual_blocks=16, num_features=64):
        super(ImageEnhancer, self).__init__()
        
        # Первый сверточный слой
        self.conv1 = nn.Conv2d(3, num_features, 9, padding=4)
        self.prelu = nn.PReLU()
        
        # Остаточные блоки
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(num_features))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Второй сверточный слой
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        # Финальный слой
        self.conv3 = nn.Conv2d(num_features, 3, 9, padding=4)
        
    def forward(self, x):
        # Сохраняем для skip-connection
        identity = x
        
        # Первый слой
        out = self.prelu(self.conv1(x))
        
        # Сохраняем для skip-connection внутри сети
        residual_out = out
        
        # Остаточные блоки
        out = self.residual_blocks(out)
        
        # Второй слой
        out = self.bn2(self.conv2(out))
        out += residual_out  # Skip connection
        
        # Финальный слой
        out = self.conv3(out)
        out += identity  # Global skip connection
        
        return out

# ===================== КОНФИГУРАЦИЯ =====================
class Config:
    # Пути
    HQ_FOLDER = 'himage'          # Папка с хорошими изображениями
    LQ_FOLDER = 'limage'          # Папка с плохими изображениями
    SAVE_DIR = 'training_results' # Папка для сохранения результатов
    
    # Параметры обучения
    BATCH_SIZE = 8
    EPOCHS = 50  
    LEARNING_RATE = 0.0001
    VAL_SPLIT = 0.2               # Доля данных для валидации
    
    # Параметры модели
    NUM_RESIDUAL_BLOCKS = 8  # для CPU
    NUM_FEATURES = 64
    
    # Сохранение
    SAVE_EVERY_EPOCH = 5     # Сохранять каждые 5 эпох для теста
    SAVE_IMAGES_EVERY = 5    # Сохранять примеры каждые 5 эпох
    
    # Другое
    SEED = 42
    NUM_WORKERS = 0  # Поставил 0 для CPU
    DEVICE = 'cpu'   # Принудительно CPU для теста

# ===================== ДАТАСЕТ =====================
class EnhancementDataset(Dataset):
    def __init__(self, hq_folder, lq_folder, transform=None, train=True):
        self.hq_folder = Path(hq_folder)
        self.lq_folder = Path(lq_folder)
        self.transform = transform
        self.train = train
        
        # Получаем список парных изображений
        self.image_pairs = []
        
        # Ищем файлы в обеих папках
        hq_files = {}
        lq_files = {}
        
        # Поддерживаемые форматы
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            # HQ файлы
            for f in self.hq_folder.glob(f'*{ext}'):
                hq_files[f.stem] = f
            for f in self.hq_folder.glob(f'*{ext.upper()}'):
                hq_files[f.stem] = f
                
            # LQ файлы
            for f in self.lq_folder.glob(f'*{ext}'):
                lq_files[f.stem] = f
            for f in self.lq_folder.glob(f'*{ext.upper()}'):
                lq_files[f.stem] = f
        
        # Находим общие файлы
        common_stems = set(hq_files.keys()) & set(lq_files.keys())
        
        for stem in common_stems:
            self.image_pairs.append((hq_files[stem], lq_files[stem]))
        
        print(f"Found {len(self.image_pairs)} image pairs")
        
        # Аугментации для тренировочного набора
        if train:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentations = None
            
        # Базовые преобразования
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hq_path, lq_path = self.image_pairs[idx]
        
        # Загружаем изображения
        hq_img = Image.open(hq_path).convert('RGB')
        lq_img = Image.open(lq_path).convert('RGB')
        
        # Проверяем размер
        if hq_img.size != (178, 218) or lq_img.size != (178, 218):
            # Изменяем размер если нужно
            hq_img = hq_img.resize((178, 218), Image.Resampling.LANCZOS)
            lq_img = lq_img.resize((178, 218), Image.Resampling.LANCZOS)
        
        # Аугментации (только для тренировочного набора)
        if self.train and self.augmentations:
            # Применяем одинаковые аугментации к обеим изображениям
            seed = np.random.randint(0, 100000)
            
            torch.manual_seed(seed)
            hq_img = self.augmentations(hq_img)
            
            torch.manual_seed(seed)
            lq_img = self.augmentations(lq_img)
        
        # Преобразуем в тензоры
        hq_tensor = self.transform(hq_img)
        lq_tensor = self.transform(lq_img)
        
        return lq_tensor, hq_tensor

# ===================== МЕТРИКИ (УПРОЩЕННЫЕ) =====================
class MetricsCalculator:
    """метрики качества"""
    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """Вычисляет PSNR (Peak Signal-to-Noise Ratio)"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr_value.item()
    
    @staticmethod
    def ssim(pred, target, window_size=11):
        
        try:
            # Простая реализация SSIM
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_x = torch.mean(pred)
            mu_y = torch.mean(target)
            sigma_x = torch.std(pred)
            sigma_y = torch.std(target)
            sigma_xy = torch.mean((pred - mu_x) * (target - mu_y))
            
            ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                       ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
            
            return ssim_val.item()
        except:
            return 0.0  # Возвращаем 0 если ошибка

# ===================== ЛОГГЕР =====================
class TrainingLogger:
    """Логирование в консоль и файл"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Файлы для логирования
        self.log_file = self.log_dir / 'training_log.txt'
        self.metrics_file = self.log_dir / 'metrics.json'
        
        # Инициализируем файлы с кодировкой UTF-8
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_psnr': [],
            'val_psnr': [],
            'epoch_times': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_psnr, val_psnr, epoch_time, lr):
        """Логирует информацию об эпохе"""
        
        # Обновляем историю
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_psnr'].append(train_psnr)
        self.metrics_history['val_psnr'].append(val_psnr)
        self.metrics_history['epoch_times'].append(epoch_time)
        
        # Форматируем сообщение
        message = (f"\n{'='*80}\n"
                  f"Epoch {epoch:03d}\n"
                  f"{'-'*40}\n"
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n"
                  f"Train PSNR: {train_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB\n"
                  f"Learning Rate: {lr:.6f}\n"
                  f"Epoch Time: {epoch_time:.1f}s\n"
                  f"{'='*80}")
        
        # Выводим в консоль
        print(message)
        
        # Записываем в файл с UTF-8
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
        
        # Сохраняем метрики в JSON
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_message(self, message):
        """Логирует произвольное сообщение"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        print(full_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + "\n")

# ===================== ВИЗУАЛИЗАЦИЯ =====================
class TrainingVisualizer:
    """Создает графики и визуализации обучения"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_metrics(self, metrics_history):
        """Создает графики метрик"""
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # График потерь
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График PSNR
        axes[0, 1].plot(epochs, metrics_history['train_psnr'], 'b-', label='Train PSNR')
        axes[0, 1].plot(epochs, metrics_history['val_psnr'], 'r-', label='Val PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График времени эпох
        axes[1, 0].plot(epochs, metrics_history['epoch_times'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Epoch Training Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Оставляем пустым для будущих метрик
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_image_comparison(self, lq_img, pred_img, hq_img, epoch, save_dir):
        """Сохраняет сравнение изображений"""
        try:
            # Конвертируем тензоры в изображения
            def tensor_to_image(tensor):
                img = tensor.cpu().detach().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                return img
            
            lq_np = tensor_to_image(lq_img)
            pred_np = tensor_to_image(pred_img)
            hq_np = tensor_to_image(hq_img)
            
            # Создаем комбинированное изображение
            comparison = np.hstack([lq_np, pred_np, hq_np])
            
            # Сохраняем
            save_path = save_dir / f'comparison_epoch_{epoch:03d}.jpg'
            cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            return save_path
        except Exception as e:
            print(f"Warning: Could not save image comparison: {e}")
            return None

# ===================== ОСНОВНОЙ КЛАСС ОБУЧЕНИЯ =====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Создаем директории
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.models_dir = self.save_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.save_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        
        # Инициализируем компоненты
        self.logger = TrainingLogger(self.save_dir)
        self.visualizer = TrainingVisualizer(self.save_dir)
        self.metrics_calc = MetricsCalculator()
        
        # Загружаем данные
        self._load_data()
        
        # Инициализируем модель
        self._init_model()
        
        # Сохраняем конфигурацию
        self._save_config()
    
    def _save_config(self):
        """Сохраняет конфигурацию в файл"""
        config_dict = {k: v for k, v in vars(self.config).items() 
                      if not k.startswith('_')}
        
        with open(self.save_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def _load_data(self):
        """Загружает и разделяет данные"""
        print("Loading data...")
        
        # Создаем полный датасет
        full_dataset = EnhancementDataset(
            hq_folder=self.config.HQ_FOLDER,
            lq_folder=self.config.LQ_FOLDER,
            train=True
        )
        
        # Разделяем на train/val
        val_size = int(len(full_dataset) * self.config.VAL_SPLIT)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Создаем DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=False
        )
        
        print(f"Data loaded:")
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val: {len(val_dataset)} images")
    
    def _init_model(self):
        """Инициализирует модель, loss и optimizer"""
        print(f"Initializing model on device: {self.device}")
        
        # Модель
        self.model = ImageEnhancer(
            num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS,
            num_features=self.config.NUM_FEATURES
        ).to(self.device)
        
        # Loss функция - используем только L1 для начала
        self.criterion = nn.L1Loss()
        
        # Оптимизатор
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Планировщик скорости обучения
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )
        
        # Счетчики параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        train_loss = 0.0
        train_psnr = 0.0
        
        for batch_idx, (lq_imgs, hq_imgs) in enumerate(self.train_loader):
            # Перемещаем на устройство
            lq_imgs = lq_imgs.to(self.device)
            hq_imgs = hq_imgs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(lq_imgs)
            
            # Вычисляем loss
            loss = self.criterion(outputs, hq_imgs)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Вычисляем метрики
            with torch.no_grad():
                batch_psnr = self.metrics_calc.psnr(outputs, hq_imgs)
            
            # Обновляем средние значения
            train_loss += loss.item()
            train_psnr += batch_psnr
            
            # Прогресс каждые 20% батчей
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                print(f"  Progress: {100*(batch_idx+1)/len(self.train_loader):.0f}%, "
                      f"Loss: {loss.item():.6f}, PSNR: {batch_psnr:.2f} dB")
        
        # Средние значения за эпоху
        avg_loss = train_loss / len(self.train_loader)
        avg_psnr = train_psnr / len(self.train_loader)
        
        return avg_loss, avg_psnr
    
    def validate(self):
        """Валидация"""
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for lq_imgs, hq_imgs in self.val_loader:
                lq_imgs = lq_imgs.to(self.device)
                hq_imgs = hq_imgs.to(self.device)
                
                outputs = self.model(lq_imgs)
                
                loss = self.criterion(outputs, hq_imgs)
                psnr = self.metrics_calc.psnr(outputs, hq_imgs)
                
                val_loss += loss.item()
                val_psnr += psnr
        
        # Средние значения
        avg_loss = val_loss / len(self.val_loader)
        avg_psnr = val_psnr / len(self.val_loader)
        
        return avg_loss, avg_psnr
    
    def save_model(self, epoch, is_best=False):
        """Сохраняет модель"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': getattr(self, 'last_train_loss', 0),
            'val_loss': getattr(self, 'last_val_loss', 0),
        }
        
        # Регулярное сохранение
        save_path = self.models_dir / f'model_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, save_path)
        
        # Лучшая модель
        if is_best:
            torch.save(checkpoint, self.models_dir / 'best_model.pth')
        
        # Последняя модель
        torch.save(checkpoint, self.models_dir / 'latest_model.pth')
        
        self.logger.log_message(f"Model saved: {save_path}")
    
    def save_example_images(self, epoch):
        """Сохраняет примеры улучшения"""
        self.model.eval()
        
        try:
            # Берем первый batch из валидации
            data_iter = iter(self.val_loader)
            lq_imgs, hq_imgs = next(data_iter)
            
            # Берем только первое изображение для экономии места
            lq_img = lq_imgs[0:1].to(self.device)
            
            with torch.no_grad():
                enhanced_img = self.model(lq_img)
            
            # Сохраняем
            save_path = self.visualizer.save_image_comparison(
                lq_imgs[0],
                enhanced_img[0],
                hq_imgs[0],
                epoch,
                self.images_dir
            )
            
            if save_path:
                self.logger.log_message(f"Example image saved: {save_path}")
        except Exception as e:
            self.logger.log_message(f"Warning: Could not save example image: {e}")
    
    def train(self):
        """Основной цикл обучения"""
        self.logger.log_message("Starting training!")
        self.logger.log_message(f"Total epochs: {self.config.EPOCHS}")
        self.logger.log_message(f"Batch size: {self.config.BATCH_SIZE}")
        self.logger.log_message(f"Learning rate: {self.config.LEARNING_RATE}")
        
        # История для лучшей модели
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Основной цикл обучения
        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.EPOCHS}")
            print(f"{'='*60}")
            
            # Обучение
            train_loss, train_psnr = self.train_epoch(epoch)
            self.last_train_loss = train_loss
            
            # Валидация
            val_loss, val_psnr = self.validate()
            self.last_val_loss = val_loss
            
            # Обновление learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Время эпохи
            epoch_time = time.time() - start_time
            
            # Логирование
            self.logger.log_epoch(
                epoch, train_loss, val_loss, 
                train_psnr, val_psnr,
                epoch_time, current_lr
            )
            
            # Сохранение каждые N эпох
            if epoch % self.config.SAVE_EVERY_EPOCH == 0:
                self.save_model(epoch)
            
            # Сохранение изображений каждые N эпох
            if epoch % self.config.SAVE_IMAGES_EVERY == 0:
                self.save_example_images(epoch)
            
            # Проверка на лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_model(epoch, is_best=True)
                self.logger.log_message(f"New best model! Val Loss: {val_loss:.6f}")
            
            # Обновление графиков каждые 5 эпох
            if epoch % 5 == 0:
                self.visualizer.plot_metrics(self.logger.metrics_history)
                self.logger.log_message("Metrics plots updated")
        
        # Финальное сохранение
        self.save_model(self.config.EPOCHS)
        self.visualizer.plot_metrics(self.logger.metrics_history)
        
        # Итоги
        self.logger.log_message("\n" + "="*80)
        self.logger.log_message("TRAINING COMPLETED!")
        self.logger.log_message(f"Best model: epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")
        self.logger.log_message(f"All results saved in: {self.save_dir}")
        
        # Сохраняем финальный отчет
        self._save_final_report(best_epoch, best_val_loss)
    
    def _save_final_report(self, best_epoch, best_val_loss):
        """Сохраняет финальный отчет"""
        report = {
            'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_epochs': self.config.EPOCHS,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'device_used': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'final_metrics': {
                'train_loss': self.logger.metrics_history['train_loss'][-1],
                'val_loss': self.logger.metrics_history['val_loss'][-1],
                'train_psnr': self.logger.metrics_history['train_psnr'][-1],
                'val_psnr': self.logger.metrics_history['val_psnr'][-1],
            }
        }
        
        with open(self.save_dir / 'final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

# ===================== ЗАПУСК ОБУЧЕНИЯ =====================
def main():
    """Основная функция запуска обучения"""
    print("\n" + "="*80)
    print("NEURAL NETWORK FOR IMAGE QUALITY ENHANCEMENT")
    print("="*80)
    
    # Проверяем данные
    if not os.path.exists(Config.HQ_FOLDER):
        print(f"\nERROR: Folder '{Config.HQ_FOLDER}' not found!")
        print(f"Create folder '{Config.HQ_FOLDER}' and put your high-quality images there")
        return
    
    if not os.path.exists(Config.LQ_FOLDER):
        print(f"\nERROR: Folder '{Config.LQ_FOLDER}' not found!")
        print(f"Create degraded images first using create_degraded_images.py")
        return
    
    # Проверяем, есть ли файлы в папках
    hq_files = [f for f in os.listdir(Config.HQ_FOLDER) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    lq_files = [f for f in os.listdir(Config.LQ_FOLDER) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(hq_files) == 0:
        print(f"\nERROR: No images found in '{Config.HQ_FOLDER}'!")
        print("Put your high-quality images in that folder")
        return
    
    if len(lq_files) == 0:
        print(f"\nERROR: No images found in '{Config.LQ_FOLDER}'!")
        print("Create degraded images first")
        return
    
    print(f"\nFound {len(hq_files)} images in '{Config.HQ_FOLDER}'")
    print(f"Found {len(lq_files)} images in '{Config.LQ_FOLDER}'")
    
    # Предупреждение о времени
    if Config.DEVICE == 'cpu':
        print("\nWARNING: Using CPU for training. This will be SLOW!")
        print("Expected training time: 30-60 minutes for 50 epochs")
        print("Consider using GPU if available.")
    
    print("\nStarting training in 5 seconds...")
    time.sleep(5)
    
    # Создаем и запускаем тренер
    trainer = Trainer(Config)
    trainer.train()

if __name__ == "__main__":
    # Устанавливаем seed для воспроизводимости
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    main()