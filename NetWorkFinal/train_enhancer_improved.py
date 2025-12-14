# train_enhancer_improved.py
import torch
import torch.nn as nn
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
from pathlib import Path

# Импортируем улучшенную архитектуру
from model_architecture_improved import ImageEnhancerImproved

# ===================== КОНФИГУРАЦИЯ =====================
class Config:
    # Пути
    HQ_FOLDER = 'himage'
    LQ_FOLDER = 'limage'
    SAVE_DIR = 'training_results_improved'
    
    # Параметры обучения
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    VAL_SPLIT = 0.2
    
    # Параметры модели
    NUM_RESIDUAL_BLOCKS = 8
    NUM_FEATURES = 64
    DROPOUT_RATE = 0.1
    
    # Сохранение
    SAVE_EVERY_EPOCH = 5
    SAVE_IMAGES_EVERY = 5
    
    # Ранняя остановка
    PATIENCE = 5
    
    # Другое
    SEED = 42
    NUM_WORKERS = 0
    DEVICE = 'cpu'

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
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in valid_extensions:
            for f in self.hq_folder.glob(f'*{ext}'):
                hq_files[f.stem] = f
            for f in self.hq_folder.glob(f'*{ext.upper()}'):
                hq_files[f.stem] = f
                
            for f in self.lq_folder.glob(f'*{ext}'):
                lq_files[f.stem] = f
            for f in self.lq_folder.glob(f'*{ext.upper()}'):
                lq_files[f.stem] = f
        
        common_stems = set(hq_files.keys()) & set(lq_files.keys())
        
        for stem in common_stems:
            self.image_pairs.append((hq_files[stem], lq_files[stem]))
        
        print(f"Found {len(self.image_pairs)} image pairs")
        
        if train:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ])
        else:
            self.augmentations = None
            
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        hq_path, lq_path = self.image_pairs[idx]
        
        hq_img = Image.open(hq_path).convert('RGB')
        lq_img = Image.open(lq_path).convert('RGB')
        
        if hq_img.size != (178, 218) or lq_img.size != (178, 218):
            hq_img = hq_img.resize((178, 218), Image.Resampling.LANCZOS)
            lq_img = lq_img.resize((178, 218), Image.Resampling.LANCZOS)
        
        if self.train and self.augmentations:
            seed = np.random.randint(0, 100000)
            
            torch.manual_seed(seed)
            hq_img = self.augmentations(hq_img)
            
            torch.manual_seed(seed)
            lq_img = self.augmentations(lq_img)
        
        hq_tensor = self.transform(hq_img)
        lq_tensor = self.transform(lq_img)
        
        return lq_tensor, hq_tensor

# ===================== МЕТРИКИ =====================
class MetricsCalculator:
    @staticmethod
    def psnr(pred, target, max_val=1.0):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr_value.item()

# ===================== ЛОГГЕР =====================
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_file = self.log_dir / 'training_log.txt'
        self.metrics_file = self.log_dir / 'metrics.json'
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
        
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_psnr': [], 'val_psnr': [],
            'epoch_times': [], 'learning_rates': []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss, train_psnr, val_psnr, epoch_time, lr):
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_psnr'].append(train_psnr)
        self.metrics_history['val_psnr'].append(val_psnr)
        self.metrics_history['epoch_times'].append(epoch_time)
        self.metrics_history['learning_rates'].append(lr)
        
        message = (f"\n{'='*80}\n"
                  f"Epoch {epoch:03d}\n"
                  f"{'-'*40}\n"
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}\n"
                  f"Train PSNR: {train_psnr:.2f} dB | Val PSNR: {val_psnr:.2f} dB\n"
                  f"Learning Rate: {lr:.6f}\n"
                  f"Epoch Time: {epoch_time:.1f}s\n"
                  f"{'='*80}")
        
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + "\n")

# ===================== ТРЕНЕР С РАННЕЙ ОСТАНОВКОЙ =====================
class ImprovedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.models_dir = self.save_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.save_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        
        self.logger = TrainingLogger(self.save_dir)
        self.metrics_calc = MetricsCalculator()
        
        self._load_data()
        self._init_model()
        self._save_config()
        
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _save_config(self):
        config_dict = {k: v for k, v in vars(self.config).items() 
                      if not k.startswith('_')}
        with open(self.save_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def _load_data(self):
        print("Loading data...")
        full_dataset = EnhancementDataset(
            hq_folder=self.config.HQ_FOLDER,
            lq_folder=self.config.LQ_FOLDER,
            train=True
        )
        
        val_size = int(len(full_dataset) * self.config.VAL_SPLIT)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
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
        
        print(f"Data loaded: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def _init_model(self):
        print(f"Initializing model on device: {self.device}")
        
        # Используем улучшенную модель с dropout
        self.model = ImageEnhancerImproved(
            num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS,
            num_features=self.config.NUM_FEATURES,
            dropout_rate=self.config.DROPOUT_RATE
        ).to(self.device)
        
        self.criterion = nn.L1Loss()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5  # L2 регуляризация
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created: {total_params:,} parameters")
    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_psnr = 0.0
        
        for batch_idx, (lq_imgs, hq_imgs) in enumerate(self.train_loader):
            lq_imgs = lq_imgs.to(self.device)
            hq_imgs = hq_imgs.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(lq_imgs)
            loss = self.criterion(outputs, hq_imgs)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            with torch.no_grad():
                batch_psnr = self.metrics_calc.psnr(outputs, hq_imgs)
            
            train_loss += loss.item()
            train_psnr += batch_psnr
            
            if (batch_idx + 1) % max(1, len(self.train_loader) // 4) == 0:
                print(f"  Progress: {100*(batch_idx+1)/len(self.train_loader):.0f}%, "
                      f"Loss: {loss.item():.6f}, PSNR: {batch_psnr:.2f} dB")
        
        return train_loss / len(self.train_loader), train_psnr / len(self.train_loader)
    
    def validate(self):
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
        
        return val_loss / len(self.val_loader), val_psnr / len(self.val_loader)
    
    def save_model(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': getattr(self, 'last_val_loss', 0),
            'val_psnr': getattr(self, 'last_val_psnr', 0),
            'config': {
                'num_residual_blocks': self.config.NUM_RESIDUAL_BLOCKS,
                'num_features': self.config.NUM_FEATURES,
                'dropout_rate': self.config.DROPOUT_RATE,
                'model_type': 'ImageEnhancerImproved'  # Добавляем информацию о типе модели
            }
        }
        
        save_path = self.models_dir / f'model_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, save_path)
        
        if is_best:
            torch.save(checkpoint, self.models_dir / 'best_model.pth')
        
        torch.save(checkpoint, self.models_dir / 'latest_model.pth')
        self.logger.log_message(f"Model saved: {save_path}")
    
    def train(self):
        self.logger.log_message("Starting training with early stopping!")
        self.logger.log_message(f"Total epochs: {self.config.EPOCHS}")
        self.logger.log_message(f"Patience: {self.config.PATIENCE} epochs")
        
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
            self.last_val_psnr = val_psnr
            
            # Обновление learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            # Логирование
            self.logger.log_epoch(
                epoch, train_loss, val_loss, 
                train_psnr, val_psnr,
                epoch_time, current_lr
            )
            
            # Проверка на лучшую модель
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch, is_best=True)
                self.logger.log_message(f"New best model! Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
                self.patience_counter = 0  # Сброс счетчика
            else:
                self.patience_counter += 1
                self.logger.log_message(f"No improvement. Patience: {self.patience_counter}/{self.config.PATIENCE}")
            
            # Сохранение каждые N эпох
            if epoch % self.config.SAVE_EVERY_EPOCH == 0:
                self.save_model(epoch)
            
            # Ранняя остановка
            if self.patience_counter >= self.config.PATIENCE:
                self.logger.log_message(f"\nEarly stopping triggered!")
                self.logger.log_message(f"No improvement for {self.config.PATIENCE} consecutive epochs.")
                self.logger.log_message(f"Best model was at epoch {self.best_epoch} with loss {self.best_val_loss:.6f}")
                break
        
        self.logger.log_message("\n" + "="*80)
        self.logger.log_message(f"Training completed! Best epoch: {self.best_epoch}")
        self.logger.log_message(f"Best Val Loss: {self.best_val_loss:.6f}")

def main():
    print("\n" + "="*80)
    print("IMPROVED TRAINING WITH EARLY STOPPING")
    print("="*80)
    print(f"Using improved model with dropout (rate={Config.DROPOUT_RATE})")
    print("="*80)
    
    # Проверка данных
    if not os.path.exists(Config.HQ_FOLDER):
        print(f"ERROR: Folder '{Config.HQ_FOLDER}' not found!")
        return
    
    if not os.path.exists(Config.LQ_FOLDER):
        print(f"ERROR: Folder '{Config.LQ_FOLDER}' not found!")
        return
    
    hq_files = [f for f in os.listdir(Config.HQ_FOLDER) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    lq_files = [f for f in os.listdir(Config.LQ_FOLDER) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nFound {len(hq_files)} images in '{Config.HQ_FOLDER}'")
    print(f"Found {len(lq_files)} images in '{Config.LQ_FOLDER}'")
    
    print("\nStarting improved training...")
    time.sleep(3)
    
    trainer = ImprovedTrainer(Config)
    trainer.train()

if __name__ == "__main__":
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    main()