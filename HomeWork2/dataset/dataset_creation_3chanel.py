import os
import torch
import tifffile as tiff

# Создадим свой класс с данными наследованием от класса Dataset
class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = [] # Список для хранения путей к файлам
        
        # Проверяем существование папки
        if not os.path.exists(root_dir):
            raise ValueError(f"Папка {root_dir} не существует!")
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):  # Проверяем что это папка
                for file in os.listdir(folder_path):
                    if file.endswith('.tif') and '_mask' not in file:
                        img_path = os.path.join(folder_path, file)
                        mask_path = img_path.replace('.tif', '_mask.tif')
                        if os.path.exists(mask_path):
                            self.samples.append((img_path, mask_path))
        
        print(f"Загружено {len(self.samples)} пар изображение-маска")
    
    def __len__(self):
        return len(self.samples) #  Длина списка с путями к файлам
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = tiff.imread(img_path).transpose(2, 0, 1)  # [3, H, W]
        mask = tiff.imread(mask_path)[None, :, :]         # [1, H, W]
        return torch.tensor(image).float(), torch.tensor(mask).float()

# Создаем датасет
dataset = BrainDataset(root_dir='./examples') 
  