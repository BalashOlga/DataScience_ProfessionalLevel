# Импорт библиотек
import torch
import os
import shutil
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import subprocess
import sys
import argparse

def setup_transfer_learning(data_yaml_path):
    """Настройка трансферного обучения для добавления 2 новых классов"""
    print("Настройка трансферного обучения для 2 новых классов...")
    
    try:
        # Загружаем конфигурацию данных
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Получаем количество классов из data.yaml
        nc = len(data_config['names'])
        print(f"Количество классов для дообучения: {nc}")
        print(f"Названия новых классов: {data_config['names']}")
        
        if nc != 2:
            print(f"⚠ Внимание: ожидается 2 класса для дообучения, но найдено {nc}")
        
        training_command = [
            sys.executable, 'yolov5/train.py',
            '--img', '640',
            '--batch', '16', 
            '--epochs', '50',
            '--data', data_yaml_path,
            '--weights', 'yolov5s.pt',
            '--freeze', '10',
            '--cache',
            '--patience', '15',
            '--project', 'results',
            '--name', 'transfer_learning',
            '--exist-ok',
        ]
        
        return training_command, nc
        
    except Exception as e:
        print(f"✗ Ошибка при настройки трансферного обучения: {e}")
        return None, None

def check_data_yaml_structure(data_yaml_path):
    """Проверяет структуру data.yaml для 2 классов"""
    print("Проверка структуры data.yaml...")
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Проверяем наличие всех необходимых полей
        required_fields = ['train', 'val', 'names']
        for field in required_fields:
            if field not in data_config:
                print(f"✗ Отсутствует обязательное поле: {field}")
                return False
        
        # Проверяем количество классов
        nc = len(data_config['names'])
        print(f"Количество классов в data.yaml: {nc}")
        
        if nc != 2:
            print(f"⚠ Внимание: ожидается 2 класса для дообучения, но найдено {nc}")
        
        # Выводим информацию о классах
        print("Классы для дообучения:")
        for i, class_name in enumerate(data_config['names']):
            print(f"  {i}. {class_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка при проверке data.yaml: {e}")
        return False

def check_and_install_requirements():
    """Проверка и установка всех необходимых зависимостей"""
    print("Проверка и установка зависимостей...")
    
    # Устанавливаем базовые зависимости
    base_requirements = [
        'torch', 'torchvision', 'torchaudio',
        'tensorboard', 'matplotlib', 'pandas', 
        'pyyaml', 'opencv-python', 'seaborn'
    ]
    
    for package in base_requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✓ {package} установлен")
        except subprocess.CalledProcessError:
            print(f"✗ Ошибка при установке {package}")
    
    # Устанавливаем зависимости из YOLOv5
    yolov5_requirements = 'yolov5/requirements.txt'
    if os.path.exists(yolov5_requirements):
        print("Установка зависимостей YOLOv5...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', yolov5_requirements], 
                         check=True)
            print("✓ Зависимости YOLOv5 установлены")
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка при установке зависимостей YOLOv5: {e}")
    else:
        print("⚠ Файл requirements.txt YOLOv5 не найден")

def setup_environment():
    """Настройка окружения и проверка зависимостей"""
    print("Проверка окружения...")
    
    # Проверка доступности GPU
    print(f"GPU доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    
    # Проверка Python
    print(f"Версия Python: {sys.version}")
    
    # Создание необходимых директорий
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

def clone_yolov5():
    """Клонирование репозитория YOLOv5"""
    print("Клонирование YOLOv5...")
    
    if not os.path.exists('yolov5'):
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'], 
                         check=True, capture_output=True)
            print("✓ YOLOv5 успешно клонирован")
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка при клонировании YOLOv5: {e}")
            sys.exit(1)
    else:
        print("✓ YOLOv5 уже существует")
    
    # Обновляем YOLOv5 если уже существует
    try:
        os.chdir('yolov5')
        subprocess.run(['git', 'pull'], check=True, capture_output=True)
        os.chdir('..')
        print("✓ YOLOv5 обновлен до последней версии")
    except subprocess.CalledProcessError:
        print("⚠ Не удалось обновить YOLOv5")

def setup_data_yaml(data_yaml_path):
    """Настройка data.yaml файла"""
    print(f"Используется YAML файл: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        print(f"✗ Ошибка: файл {data_yaml_path} не найден!")
        sys.exit(1)
    
    # Проверка и исправление data.yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Исправление test пути если он пустой
        if not data_config.get('test'):
            data_config['test'] = data_config['val']  # используем val для теста
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print("✓ data.yaml успешно проверен и обновлен")
        return data_config
        
    except Exception as e:
        print(f"✗ Ошибка при обработке data.yaml: {e}")
        sys.exit(1)

def start_tensorboard():
    """Запуск TensorBoard в фоновом режиме"""
    print("Запуск TensorBoard...")
    try:
        # Создаем логи директорию для TensorBoard
        os.makedirs('results/training', exist_ok=True)
        
        # Запускаем TensorBoard
        tensorboard_process = subprocess.Popen([
            sys.executable, '-m', 'tensorboard.main',
            '--logdir', 'results',
            '--port', '6006',
            '--host', 'localhost'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✓ TensorBoard запущен на http://localhost:6006")
        print("  Для остановки TensorBoard закройте это окно или нажмите Ctrl+C")
        return tensorboard_process
        
    except Exception as e:
        print(f"✗ Ошибка при запуске TensorBoard: {e}")
        return None

def train_model(training_command):
    """Обучение модели YOLOv5"""
    print("Запуск дообучения YOLOv5 на 2 новых классах...")
    
    try:
        print("Команда обучения:", ' '.join(training_command))
        result = subprocess.run(training_command, check=True)
        print("✓ Дообучение завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при дообучении модели: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Дообучение прервано пользователем")
        return False

def test_model(data_yaml_path):
    """Тестирование дообученной модели"""
    print("Тестирование дообученной модели...")
    
    best_model_path = 'results/transfer_learning/weights/best.pt'
    if not os.path.exists(best_model_path):
        print(f"✗ Дообученная модель не найдена: {best_model_path}")
        return False
    
    test_command = [
        sys.executable, 'yolov5/val.py',
        '--weights', best_model_path,
        '--data', data_yaml_path,
        '--img', '640',
        '--batch', '16',
        '--task', 'val',
        '--project', 'results',
        '--name', 'test_transfer',
        '--exist-ok'
    ]
    
    try:
        subprocess.run(test_command, check=True)
        print("✓ Тестирование дообученной модели завершено")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при тестировании модели: {e}")
        return False

def copy_all_files(source_dir, dest_dir):
    """Копирование всех файлов из source_dir в dest_dir"""
    if os.path.exists(source_dir):
        all_files = glob.glob(f'{source_dir}/*')
        for file_path in all_files:
            if os.path.isfile(file_path):
                shutil.copy(file_path, dest_dir)
                print(f"  Скопирован: {os.path.basename(file_path)}")

def save_results():
    """Сохранение и копирование результатов"""
    print("Копирование результатов дообучения...")
    
    # Копирование графиков и файлов
    if os.path.exists('results/transfer_learning'):
        copy_all_files('results/transfer_learning', 'results/plots')
    
    if os.path.exists('results/test_transfer'):
        copy_all_files('results/test_transfer', 'results/plots')
    
    # Копирование моделей
    best_model_path = 'results/transfer_learning/weights/best.pt'
    last_model_path = 'results/transfer_learning/weights/last.pt'
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'results/models/best_transfer.pt')
        print("✓ Дообученная модель скопирована")
    
    if os.path.exists(last_model_path):
        shutil.copy(last_model_path, 'results/models/last_transfer.pt')
        print("✓ Последняя версия дообученной модели скопирована")

def create_plots():
    """Создание графиков метрик дообучения"""
    print("Создание графиков метрик дообучения...")
    
    # Поиск файлов с результатами
    results_files = []
    
    # Ищем CSV файлы (новый формат)
    csv_files = glob.glob('results/transfer_learning/*.csv')
    if csv_files:
        results_files.extend(csv_files)
    
    # Ищем текстовые файлы (старый формат)
    txt_files = glob.glob('results/transfer_learning/results*.txt')
    if txt_files:
        results_files.extend(txt_files)
    
    if not results_files:
        print("✗ Не найдены файлы с результатами дообучения")
        return None, None, None, None, None, False
    
    results_file = results_files[0]
    print(f"Используется файл: {results_file}")
    
    epochs = []
    train_loss = []
    val_loss = []
    precision = []
    recall = []
    map50 = []
    map5095 = []
    
    try:
        if results_file.endswith('.csv'):
            # Чтение CSV файла (новый формат YOLOv5)
            df = pd.read_csv(results_file)
            if not df.empty:
                epochs = list(range(1, len(df) + 1))
                
                # Маппинг колонок для нового формата
                column_mapping = {
                    'train/box_loss': train_loss,
                    'val/box_loss': val_loss, 
                    'metrics/precision(B)': precision,
                    'metrics/recall(B)': recall,
                    'metrics/mAP_0.5(B)': map50,
                    'metrics/mAP_0.5:0.95(B)': map5095
                }
                
                for col in df.columns:
                    for pattern, target_list in column_mapping.items():
                        if pattern in col:
                            target_list.extend(df[col].tolist())
                            print(f"  Найдена колонка: {col}")
                            break
        else:
            # Парсинг текстового файла (старый формат)
            with open(results_file, 'r') as f:
                for line in f:
                    if 'epoch' in line and '/50' in line:
                        parts = line.strip().split()
                        try:
                            epoch = int(parts[1])
                            train_box_loss = float(parts[3])
                            val_box_loss = float(parts[9])
                            prec = float(parts[11])
                            rec = float(parts[12])
                            m50 = float(parts[13])
                            m5095 = float(parts[14])
                            
                            epochs.append(epoch)
                            train_loss.append(train_box_loss)
                            val_loss.append(val_box_loss)
                            precision.append(prec)
                            recall.append(rec)
                            map50.append(m50)
                            map5095.append(m5095)
                            
                        except (ValueError, IndexError):
                            continue
        
        if epochs:
            # Создание графиков
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # График потерь
            ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2, color='blue')
            ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2, color='red')
            ax1.set_title('Transfer Learning: Training vs Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # График Precision и Recall
            if precision and recall:
                ax2.plot(epochs, precision, label='Precision', linewidth=2, color='green')
                ax2.plot(epochs, recall, label='Recall', linewidth=2, color='orange')
                ax2.set_title('Precision & Recall')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Score')
                ax2.legend()
                ax2.grid(True)
            
            # График mAP
            if map50 and map5095:
                ax3.plot(epochs, map50, label='mAP@0.5', linewidth=2, color='purple')
                ax3.plot(epochs, map5095, label='mAP@0.5:0.95', linewidth=2, color='brown')
                ax3.set_title('mAP Metrics')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('mAP')
                ax3.legend()
                ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/plots/transfer_learning_metrics.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Поиск лучших метрик
            if map50:
                best_epoch_idx = map50.index(max(map50))
                best_epoch = epochs[best_epoch_idx]
                best_map50 = max(map50)
                best_map5095 = map5095[best_epoch_idx] if map5095 else 0
                best_precision = precision[best_epoch_idx] if precision else 0
                best_recall = recall[best_epoch_idx] if recall else 0
                
                print(f"✓ Лучшая эпоха: {best_epoch}")
                print(f"✓ Лучший mAP@0.5: {best_map50:.3f}")
                print(f"✓ Лучший mAP@0.5:0.95: {best_map5095:.3f}")
                print(f"✓ Precision: {best_precision:.3f}")
                print(f"✓ Recall: {best_recall:.3f}")
                
                return best_epoch, best_map50, best_map5095, best_precision, best_recall, True
            
        else:
            print("✗ Не удалось извлечь данные для графиков")
            return None, None, None, None, None, False
            
    except Exception as e:
        print(f"✗ Ошибка при создании графиков: {e}")
        return None, None, None, None, None, False

def create_report(best_epoch, best_map50, best_map5095, best_precision, best_recall, data_yaml_path):
    """Создание отчета о дообучении"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    
    report_content = f"""
ОТЧЕТ О ДООБУЧЕНИИ YOLOv5
==========================

Дата: {current_time}
GPU: {gpu_info}
Файл конфигурации: {data_yaml_path}

ЦЕЛЬ ДООБУЧЕНИЯ:
----------------
Добавление {len(class_names)} новых классов к предобученной модели YOLOv5s (80 классов COCO)

НОВЫЕ КЛАССЫ:
-------------
{chr(10).join([f"{i}. {name}" for i, name in enumerate(class_names)])}

РЕЗУЛЬТАТЫ ДООБУЧЕНИЯ:
----------------------
Лучшая эпоха: {best_epoch if best_epoch else 'N/A'}
mAP@0.5: {best_map50 if best_map50 else 'N/A':.3f}
mAP@0.5:0.95: {best_map5095 if best_map5095 else 'N/A':.3f}
Precision: {best_precision if best_precision else 'N/A':.3f}
Recall: {best_recall if best_recall else 'N/A':.3f}

ПАРАМЕТРЫ ДООБУЧЕНИЯ:
---------------------
Метод: Transfer Learning с заморозкой слоев
Замороженные слои: 10
Эпохи: 50
Размер изображения: 640
Batch size: 16
Learning rate: 0.001
"""
    
    with open('results/logs/transfer_learning_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(report_content)
    return report_content

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='YOLOv5 Transfer Learning Script')
    parser.add_argument('--data-yaml', type=str, required=True, 
                       help='Path to data.yaml file with 2 new classes')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv5 ДООБУЧЕНИЕ на 2 НОВЫХ КЛАССА")
    print("=" * 60)
    
    # Установка зависимостей (если не пропущено)
    if not args.skip_deps:
        check_and_install_requirements()
    else:
        print("⚠ Установка зависимостей пропущена")
    
    # Настройка окружения
    setup_environment()
    
    # Клонирование YOLOv5
    clone_yolov5()
    
    # Проверка структуры data.yaml
    if not check_data_yaml_structure(args.data_yaml):
        print("✗ Проверка data.yaml не пройдена")
        sys.exit(1)
    
    # Настройка data.yaml
    data_config = setup_data_yaml(args.data_yaml)
    
    # Настройка трансферного обучения
    training_command, nc = setup_transfer_learning(args.data_yaml)
    if not training_command:
        print("✗ Ошибка настройки трансферного обучения")
        sys.exit(1)
    
    # Запуск TensorBoard (опционально)
    tensorboard_process = None
    if not args.no_tensorboard:
        tensorboard_process = start_tensorboard()
    
    try:
        # Дообучение модели
        success = train_model(training_command)
        
        if success:
            # Тестирование дообученной модели
            test_model(args.data_yaml)
            
            # Сохранение результатов
            save_results()
            
            # Создание графиков
            best_epoch, best_map50, best_map5095, best_precision, best_recall, plots_created = create_plots()
            
            # Создание отчета
            create_report(best_epoch, best_map50, best_map5095, best_precision, best_recall, args.data_yaml)
            
            print("\n" + "="*50)
            print("🎉 ДООБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print("="*50)
            print("\n📁 Результаты сохранены в папке 'results/'")
            print("🤖 Дообученная модель: results/models/best_transfer.pt")
            print("📊 Графики: results/plots/")
            print("📝 Логи: results/logs/")
            
            if tensorboard_process:
                print(f"\n📈 TensorBoard доступен по адресу: http://localhost:6006")
            
            print(f"\n💡 Модель теперь распознает 80 классов COCO + {nc} новых классов!")
        
        else:
            print("\n Дообучение завершилось с ошибками")
    
    except KeyboardInterrupt:
        print("\n Дообучение прервано пользователем")
    except Exception as e:
        print(f"\n Произошла ошибка: {e}")
    finally:
        # Завершение TensorBoard
        if tensorboard_process:
            tensorboard_process.terminate()
            print("✓ TensorBoard остановлен")

if __name__ == "__main__":
    main()