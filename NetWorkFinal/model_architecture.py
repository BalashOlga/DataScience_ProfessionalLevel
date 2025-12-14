# model_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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