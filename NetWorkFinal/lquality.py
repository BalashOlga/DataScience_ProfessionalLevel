import cv2
import numpy as np
import os

def degrade_single_image(image_cv):
    """
    Ухудшает качество одного изображения 
    image_cv: изображение в формате OpenCV (BGR)
    возвращает: ухудшенное изображение в формате OpenCV (BGR)
    """
    if image_cv is None:
        return None
    
    height, width = image_cv.shape[:2]
    degraded = image_cv.copy()
    
    # 1. Уменьшаем разрешение и увеличиваем обратно
    small_width = width // 2
    small_height = height // 2
    small = cv2.resize(degraded, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    degraded = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 2. Добавляем размытие (blur)
    degraded = cv2.GaussianBlur(degraded, (5, 5), 1.0)
    
    # 3. Добавляем шум (noise)
    noise = np.random.normal(0, 10, degraded.shape).astype(np.int16)
    degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 4. Добавляем артефакты сжатия JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    success, encoded_img = cv2.imencode('.jpg', degraded, encode_params)
    if success:
        degraded = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # 5. Уменьшаем контраст
    degraded = cv2.convertScaleAbs(degraded, alpha=0.9, beta=5)
    
    return degraded

def create_degraded_versions():
    """
    Создает ухудшенные версии всех изображений из папки himage
    и сохраняет их в папку limage с теми же именами
    """
    
    print("Создание ухудшенных изображений для обучения нейросети")
    print("=" * 60)
    
    # Проверяем существование папки с исходными изображениями
    if not os.path.exists('himage'):
        print("ОШИБКА: Папка 'himage' не найдена!")
        print("\nСоздайте папку 'himage' и поместите туда:")
        print("- Фотографии размером 178x218 пикселей")
        print("- Форматы: JPG, PNG, BMP")
        print("- Затем запустите программу снова")
        return
    
    # Создаем папку для ухудшенных изображений
    os.makedirs('limage', exist_ok=True)
    
    # Получаем список всех изображений
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir('himage'):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in valid_extensions):
            image_files.append(file)
    
    if len(image_files) == 0:
        print("В папке 'himage' не найдено изображений!")
        print("Добавьте фотографии в формате JPG, PNG или BMP")
        return
    
    print(f"Найдено {len(image_files)} изображений в папке 'himage'")
    print("Начинаю создание ухудшенных версий...")
    
    # Счетчики
    processed = 0
    skipped = 0
    
    # Обрабатываем каждое изображение
    for filename in image_files:
        input_path = os.path.join('himage', filename)
        output_path = os.path.join('limage', filename)
        
        try:
            # Загружаем изображение
            img = cv2.imread(input_path)
            if img is None:
                print(f"Ошибка чтения: {filename}")
                skipped += 1
                continue
            
            # Проверяем размер
            height, width = img.shape[:2]
            
            if width != 178 or height != 218:
                print(f"Пропускаю {filename}: размер {width}x{height} (требуется 178x218)")
                skipped += 1
                continue
            
            # Используем функцию degrade_single_image для ухудшения
            degraded = degrade_single_image(img)
            
            # Сохраняем ухудшенное изображение
            cv2.imwrite(output_path, degraded, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            processed += 1
            print(f"Создано: {filename}")
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {str(e)}")
            skipped += 1
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Успешно обработано: {processed} изображений")
    print(f"Пропущено: {skipped} изображений")
    
    if processed > 0:
        print(f"\nУхудшенные изображения сохранены в папку 'limage'")
        print("\nТеперь у вас есть пары для обучения нейросети:")
        print("himage/photo1.jpg  (хорошее качество)")
        print("limage/photo1.jpg  (ухудшенное качество)")
        print("\nНейросеть будет учиться преобразовывать плохие изображения в хорошие!")
    else:
        print("\nНе удалось обработать ни одного изображения.")

def show_example():
    """Показывает пример ухудшения (оригинальная функция)"""
    if not os.path.exists('limage'):
        print("Сначала создайте ухудшенные изображения!")
        return
    
    # Ищем первое изображение в limage
    limage_files = [f for f in os.listdir('limage') if f.lower().endswith(('.jpg', '.png'))]
    
    if not limage_files:
        print("В папке limage нет изображений")
        return
    
    example_file = limage_files[0]
    himage_path = os.path.join('himage', example_file)
    limage_path = os.path.join('limage', example_file)
    
    if os.path.exists(himage_path) and os.path.exists(limage_path):
        print(f"\nПример ухудшения для файла: {example_file}")
        print("Слева: оригинал (himage), Справа: ухудшенное (limage)")
        
        # Загружаем оба изображения
        original = cv2.imread(himage_path)
        degraded = cv2.imread(limage_path)
        
        # Конвертируем BGR в RGB для правильного отображения
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        
        # Вычисляем разницу (просто для информации)
        diff = cv2.absdiff(original, degraded)
        diff_percentage = np.mean(diff) / 255.0 * 100
        
        print(f"Средняя разница между изображениями: {diff_percentage:.1f}%")
        
        # Сохраняем пример сравнения
        combined = np.hstack([original, degraded])
        cv2.imwrite('comparison_example.jpg', combined)
        print("Пример сравнения сохранен как 'comparison_example.jpg'")

if __name__ == "__main__":
    # Основной процесс
    create_degraded_versions()
    
    # Показываем пример, если что-то создано
    if os.path.exists('limage') and len(os.listdir('limage')) > 0:
        show_example()
    
    print("\nГотово! Теперь можно приступать к обучению нейросети.")