import tifffile as tiff
import os

# Быстрая проверка первых 5 файлов, чтобы понять какие данные у нас 
root_dir = './examples'
count = 0

for folder in os.listdir(root_dir):
    if count >= 5:  # Проверим только 5 файлов
        break
    folder_path = os.path.join(root_dir, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.tif') and '_mask' not in file:
            img_path = os.path.join(folder_path, file)
            mask_path = img_path.replace('.tif', '_mask.tif')
            
            image = tiff.imread(img_path)
            mask = tiff.imread(mask_path)
            
            print(f"Файл: {file}")
            print(f"Изображение: shape {image.shape}, dtype {image.dtype}")
            print(f"Маска: shape {mask.shape}, dtype {mask.dtype}")
            print("---")
            
            count += 1
            if count >= 5:
                break
    if count >= 5:
        break