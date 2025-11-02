import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from PIL import Image, ImageEnhance
import io
import sqlite3
import json
import os
from datetime import datetime
import pickle
import joblib
import tempfile

# Очиста кэша Streamlit при перезапуске приложения
st.cache_data.clear()
st.cache_resource.clear()

# Создание папок для проекта с защитой от повторного создания
PROJECT_FOLDERS = ['models', 'experiments', 'database']

def setup_project_folders():
    """Создание папок проекта если они не существуют"""
    created_folders = []
    existing_folders = []
    
    for folder in PROJECT_FOLDERS:
        try:
            os.makedirs(folder, exist_ok=True)
            # Проверяем, была ли папка создана или уже существовала
            if not os.listdir(folder):  # Папка пустая
                created_folders.append(folder)
            else:
                existing_folders.append(folder)
        except Exception as e:
            st.error(f"Ошибка при создании папки {folder}: {e}")
    
    return created_folders, existing_folders

# Инициализация папок проекта
created, existing = setup_project_folders()

# Настройка страницы
st.set_page_config(page_title="CIFAR-10 Классификатор", layout="wide")

# Настройки сохранения моделей
SAVE_SETTINGS = {
    'knn': {
        'enabled': True,
        'max_size_mb': 10
    },
    'neural_network': {
        'enabled': True, 
        'max_size_mb': 50
    },
    'cnn': {
        'enabled': True,
        'max_size_mb': 200
    }
}

# Функция для проверки размера модели
def get_model_size(model, model_type):
    """Оценивает размер модели в MB"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            if model_type in ['cnn', 'neural_network']:
                model.save(tmp.name)
            elif model_type == 'knn':
                with open(tmp.name, 'wb') as f:
                    pickle.dump(model, f)
            
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            os.unlink(tmp.name)
        
        return size_mb
    except Exception as e:
        st.error(f"Ошибка при оценке размера модели: {e}")
        return 0

# Функция для сохранения модели с проверкой
def save_model_with_check(model, filepath, model_type, experiment_name):
    """Сохраняет модель с проверкой настроек и размера"""
    
    settings = SAVE_SETTINGS.get(model_type, {'enabled': False, 'max_size_mb': 0})
    
    if not settings['enabled']:
        return False, f"Сохранение моделей {model_type} отключено в настройках"
    
    # Проверяем размер модели
def get_model_size(model, model_type):
    """Оценивает размер модели в MB с защитой от блокировки файлов"""
    try:
        # Создаем временную папку вместо файла
        with tempfile.TemporaryDirectory() as temp_dir:
            if model_type in ['cnn', 'neural_network']:
                model_path = os.path.join(temp_dir, 'model.h5')
                model.save(model_path)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
            elif model_type == 'knn':
                model_path = os.path.join(temp_dir, 'model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
            else:
                size_mb = 0
        return size_mb
    except Exception as e:
        st.error(f"Ошибка при оценке размера модели: {e}")
        return 0
    
    # Сохраняем модель
    try:
        if model_type in ['cnn', 'neural_network']:
            model.save(filepath)
        elif model_type == 'knn':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        # Проверяем что файл действительно создан
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath) / (1024 * 1024)
            return True, f"Модель сохранена ({actual_size:.1f}MB)"
        else:
            return False, "Файл модели не был создан"
    except Exception as e:
        return False, f"Ошибка сохранения: {str(e)}"

# Инициализация базы данных
def init_database():
    """Инициализация базы данных"""
    db_file = 'database/experiments.db'
    db_exists = os.path.exists(db_file)
    
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    
    # Создаем таблицу если не существует
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_filename TEXT NOT NULL,
            parameters TEXT NOT NULL,
            dataset_info TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            training_time REAL NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            enhancement_applied BOOLEAN NOT NULL,
            augmentation_type TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            model_size_mb REAL,
            save_status TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    
    # Проверяем, была ли таблица только что создана
    c.execute("SELECT COUNT(*) FROM experiments")
    record_count = c.fetchone()[0]
    
    conn.close()
    
    return db_exists, record_count

# Инициализация базы данных при запуске
db_existed, initial_records = init_database()

# Инициализация session state
if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'enhancement_applied' not in st.session_state:
    st.session_state.enhancement_applied = False
if 'x_train_enhanced' not in st.session_state:
    st.session_state.x_train_enhanced = None
if 'x_test_enhanced' not in st.session_state:
    st.session_state.x_test_enhanced = None

# Загрузка данных CIFAR-10 с кэшированием
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# Функции для улучшения качества изображений
def enhance_image(image):
    """Улучшение качества изображения"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2) # Увеличивает контраст на 20%
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3) # Увеличивает резкость на 30%
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1) # Увеличивает яркость на 10%
    
    return np.array(image) / 255.0

def get_augmentation_parameters(augmentation_type='basic'):
    """Возвращает параметры аугментации для Keras"""
    if augmentation_type == 'none':
        return None
    
    params = {
        'rotation_range': 15 if augmentation_type == 'basic' else 20,
        'width_shift_range': 0.1 if augmentation_type == 'basic' else 0.15,
        'height_shift_range': 0.1 if augmentation_type == 'basic' else 0.15,
        'horizontal_flip': True,
        'zoom_range': 0.1 if augmentation_type == 'basic' else 0.2,
        'fill_mode': 'nearest'
    }
    return params

# Функция для поиска оптимального K (метод локтя)
def find_optimal_k(x_train_flat, y_train, max_k=15):
    """Нахождение оптимального K методом локтя"""
    st.info("Ищем оптимальное K методом локтя...")
    
    k_range = range(1, max_k + 1, 2)
    k_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, k in enumerate(k_range):
        status_text.text(f"Проверяем K={k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train_flat[:2000], y_train[:2000], cv=3, scoring='accuracy')
        k_scores.append(scores.mean())
        progress_bar.progress((i + 1) / len(k_range))
    
    differences = [k_scores[i] - k_scores[i-1] for i in range(1, len(k_scores))]
    optimal_k_index = differences.index(max(differences)) + 1
    optimal_k = k_range[optimal_k_index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, k_scores, 'bo-', alpha=0.7)
    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Оптимальное K={optimal_k}')
    ax.set_xlabel('Количество соседей (K)')
    ax.set_ylabel('Точность')
    ax.set_title('Метод локтя для выбора оптимального K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return optimal_k, fig

# Функция для рекомендации параметров нейронной сети
def recommend_nn_parameters(sample_size):
    """Рекомендация параметров нейронной сети на основе размера выборки"""
    if sample_size <= 5000:
        return {'epochs': 10, 'batch_size': 32, 'units1': 64, 'units2': 32, 'dropout': 0.3}
    elif sample_size <= 15000:
        return {'epochs': 15, 'batch_size': 64, 'units1': 128, 'units2': 64, 'dropout': 0.4}
    else:
        return {'epochs': 20, 'batch_size': 128, 'units1': 256, 'units2': 128, 'dropout': 0.5}

# Функция для рекомендации параметров CNN
def recommend_cnn_parameters(sample_size):
    """Рекомендация параметров CNN на основе размера выборки"""
    if sample_size <= 5000:
        return {'epochs': 10, 'batch_size': 32, 'filters1': 32, 'filters2': 64, 'dense_units': 64}
    elif sample_size <= 15000:
        return {'epochs': 15, 'batch_size': 64, 'filters1': 64, 'filters2': 128, 'dense_units': 128}
    else:
        return {'epochs': 20, 'batch_size': 128, 'filters1': 128, 'filters2': 256, 'dense_units': 256}

# Функция для сохранения эксперимента в базу данных
def save_experiment_to_db(experiment_data):
    if experiment_data['model_type'] == 'K-NN':
        experiment_data['augmentation_type'] = 'none'

    conn = sqlite3.connect('database/experiments.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO experiments 
        (experiment_name, model_type, model_filename, parameters, dataset_info, 
         accuracy, precision, recall, f1_score, training_time, 
         enhancement_applied, augmentation_type, sample_size, model_size_mb, save_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        experiment_data['experiment_name'],
        experiment_data['model_type'],
        experiment_data['model_filename'],
        experiment_data['parameters'],
        experiment_data['dataset_info'],
        experiment_data['accuracy'],
        experiment_data['precision'],
        experiment_data['recall'],
        experiment_data['f1_score'],
        experiment_data['training_time'],
        experiment_data['enhancement_applied'],
        experiment_data['augmentation_type'],
        experiment_data['sample_size'],
        experiment_data.get('model_size_mb', 0),
        experiment_data.get('save_status', 'unknown')
    ))
    
    conn.commit()
    conn.close()

# Выгрузка данны из БД в csv
def export_experiments_to_csv():
    conn = sqlite3.connect('database/experiments.db')
    df = pd.read_sql_query('SELECT * FROM experiments', conn)
    conn.close()
    return df

# СохранЕние модели с проверкой настроек и размера
def save_model_with_check(model, filepath, model_type, experiment_name):
    try:
        settings = SAVE_SETTINGS.get(model_type, {'enabled': False, 'max_size_mb': 0})
        
        if not settings['enabled']:
            return False, f"Сохранение моделей {model_type} отключено в настройках"
        
        # Для KNN пропускаем проверку размера (они обычно маленькие)
        if model_type == 'knn':
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                actual_size = os.path.getsize(filepath) / (1024 * 1024)
                return True, f"Модель сохранена ({actual_size:.1f}MB)"
            except Exception as e:
                return False, f"Ошибка сохранения: {str(e)}"
        
        # Для остальных моделей проверяем размер
        size_mb = get_model_size(model, model_type)
        max_size_mb = settings['max_size_mb']
        
        if size_mb > max_size_mb:
            return False, f"Модель слишком большая: {size_mb:.1f}MB > {max_size_mb}MB"
        
        # Сохраняем модель
        try:
            if model_type in ['cnn', 'neural_network']:
                model.save(filepath)
            elif model_type == 'knn':
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            # Проверяем что файл создан
            if os.path.exists(filepath):
                actual_size = os.path.getsize(filepath) / (1024 * 1024)
                return True, f"Модель сохранена ({actual_size:.1f}MB)"
            else:
                return False, "Файл модели не был создан"
                
        except Exception as e:
            return False, f"Ошибка сохранения: {str(e)}"
    
    except Exception as e:
        # ГАРАНТИРОВАННЫЙ возврат при любой ошибке
        return False, f"Неожиданная ошибка: {str(e)}"

# Функция для загрузки экспериментов из базы данных
def load_experiments_from_db():
    conn = sqlite3.connect('database/experiments.db')
    df = pd.read_sql_query('SELECT * FROM experiments ORDER BY created_date DESC', conn)
    conn.close()
    return df

# Функция для создания расширенных графиков
def create_extended_plots(y_true, y_pred, y_pred_proba=None, class_names=None):
    """Создание расширенных графиков для анализа модели"""
    plots = {}
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_xlabel('Предсказанные метки')
    ax1.set_ylabel('Истинные метки')
    ax1.set_title('Матрица ошибок')
    plots['confusion_matrix'] = fig1
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    class_metrics = metrics_df.iloc[:-3, :3]
    x = range(len(class_metrics))
    width = 0.25
    
    ax2.bar([i - width for i in x], class_metrics['precision'], width, label='Precision', alpha=0.8)
    ax2.bar(x, class_metrics['recall'], width, label='Recall', alpha=0.8)
    ax2.bar([i + width for i in x], class_metrics['f1-score'], width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('Классы')
    ax2.set_ylabel('Значение метрик')
    ax2.set_title('Метрики по классам')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_metrics.index, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plots['class_metrics'] = fig2
    
    if y_pred_proba is not None:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        n_classes = len(class_names)
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        ax3.plot([0, 1], [0, 1], 'k--', lw=2)
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC-кривые по классам')
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)
        plots['roc_curves'] = fig3
    
    return plots, metrics_df

# Функция для проверки статуса проекта
def check_project_structure():
    """Проверка структуры проекта"""
    folders = ['models', 'experiments', 'database']
    status = {}
    
    for folder in folders:
        exists = os.path.exists(folder)
        is_dir = os.path.isdir(folder) if exists else False
        file_count = 0
        
        if exists and is_dir:
            try:
                file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            except:
                file_count = 0
        
        status[folder] = {
            'exists': exists,
            'is_directory': is_dir,
            'file_count': file_count
        }
    
    return status

# Названия классов
class_names = ['самолёт', 'автомобиль', 'птица', 'кошка', 'олень', 
               'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']

# Заголовок приложения
st.title("🖼️ Улучшенный классификатор изображений CIFAR-10")
st.write("Интеллектуальный подбор параметров и улучшение качества изображений")

# Показываем статус инициализации
if created:
    st.sidebar.success(f"✅ Созданы папки: {', '.join(created)}")
if existing:
    st.sidebar.info(f"📁 Используются существующие: {', '.join(existing)}")

# Загрузка данных
with st.spinner('Загружаем данные CIFAR-10...'):
    (x_train, y_train), (x_test, y_test) = load_data()

# Настройки в боковой панели
st.sidebar.title("Настройки сохранения моделей")

SAVE_SETTINGS['knn']['enabled'] = st.sidebar.checkbox("Сохранять KNN модели", value=True)
SAVE_SETTINGS['neural_network']['enabled'] = st.sidebar.checkbox("Сохранять Neural Network модели", value=True)
SAVE_SETTINGS['cnn']['enabled'] = st.sidebar.checkbox("Сохранять CNN модели", value=True)

SAVE_SETTINGS['neural_network']['max_size_mb'] = st.sidebar.slider(
    "Макс. размер NN моделей (MB)", 10, 100, 50
)
SAVE_SETTINGS['cnn']['max_size_mb'] = st.sidebar.slider(
    "Макс. размер CNN моделей (MB)", 50, 500, 200
)

# Настройки улучшения качества
st.sidebar.title("Настройки улучшения качества")
enhance_quality = st.sidebar.checkbox("Улучшить качество изображений", value=False)

# Кнопка для применения улучшения качества
if enhance_quality and not st.session_state.enhancement_applied:
    if st.sidebar.button("Применить улучшения к данным"):
        with st.spinner('Применяем улучшения к изображениям...'):
            x_train_enhanced = x_train.copy()
            x_test_enhanced = x_test.copy()
            
            st.info("🔧 Улучшаем качество изображений...")
            x_train_enhanced = np.array([enhance_image(img) for img in x_train_enhanced])
            x_test_enhanced = np.array([enhance_image(img) for img in x_test_enhanced])
            
            st.session_state.x_train_enhanced = x_train_enhanced
            st.session_state.x_test_enhanced = x_test_enhanced
            st.session_state.enhancement_applied = True
            
            st.sidebar.success("✅ Применено улучшение качества")

st.sidebar.subheader("Статус улучшения")
if st.session_state.enhancement_applied:
    st.sidebar.success("✅ Улучшение применено")
else:
    st.sidebar.info("ℹ️ Оригинальные изображения")

# Настройки аугментации
st.sidebar.title("Настройки аугментации")
enable_augmentation = st.sidebar.checkbox("Включить аугментацию данных", value=False)

if enable_augmentation:
    augmentation_type = st.sidebar.selectbox("Тип аугментации", ['basic', 'advanced'], index=0)
    
    with st.sidebar.expander("📊 Параметры аугментации"):
        aug_params = get_augmentation_parameters(augmentation_type)
        st.write("**Текущие параметры:**")
        for param, value in aug_params.items():
            st.write(f"- {param}: {value}")
else:
    augmentation_type = 'none'

st.sidebar.subheader("Статус аугментации")
if enable_augmentation:
    st.sidebar.success(f"✅ Аугментация: {augmentation_type}")
else:
    st.sidebar.info("ℹ️ Аугментация отключена")    

# Сброс улучшения при изменении настроек
if st.session_state.enhancement_applied and not enhance_quality:
    st.session_state.enhancement_applied = False
    st.session_state.x_train_enhanced = None
    st.session_state.x_test_enhanced = None
    st.sidebar.info("Улучшения отключены")

# Улучшенные настройки изменения размера выборки
st.subheader("🎛️ Настройки выборки данных")

col1, col2 = st.columns(2)

with col1:
    sample_size = st.slider(
        'Размер ТРЕНИРОВОЧНОЙ выборки', 
        1000, 50000, 10000,
        help="CIFAR-10 содержит 50,000 тренировочных изображений"
    )

with col2:
    test_sample_size = st.slider(
        'Размер ТЕСТОВОЙ выборки', 
        1000, 10000, 2000,
        help="CIFAR-10 содержит 10,000 тестовых изображений"
    )

st.info("""
**📝 Пояснение по размерам выборки:**
- **Полный датасет CIFAR-10**: 60,000 изображений (50,000 тренировочных + 10,000 тестовых)
- **Тренировочные данные**: до 50,000 изображений  
- **Тестовые данные**: до 10,000 изображений
""")    

# Подготовка данных
if st.session_state.enhancement_applied:
    x_train_small = st.session_state.x_train_enhanced[:sample_size]
    x_test_small = st.session_state.x_test_enhanced[:test_sample_size]
    current_data_type = "улучшенные"
else:
    x_train_small = x_train[:sample_size]
    x_test_small = x_test[:test_sample_size]
    current_data_type = "оригинальные"

y_train_small = y_train[:sample_size].flatten()
y_test_small = y_test[:test_sample_size].flatten()

st.success(f"Данные подготовлены! {len(x_train_small)} тренировочных и {len(x_test_small)} тестовых {current_data_type} изображений")

# Боковая панель для навигации
st.sidebar.title("Навигация")
section = st.sidebar.radio("Выберите раздел:", 
                           ["Обзор данных", "K-Nearest Neighbors", "Нейронная сеть", 
                            "Свёрточная сеть (CNN)", "Анализ результатов", "Сравнение экспериментов"])

# Раздел 1: Обзор данных
if section == "Обзор данных":
    st.header("📊 Обзор данных CIFAR-10")
    
    # Статус улучшения
    if st.session_state.enhancement_applied:
        st.success("✅ Используются улучшенные изображения")
    else:
        st.info("ℹ️ Используются оригинальные изображения")
    
    # Статус аугментации
    if enable_augmentation:
        st.info(f"🎯 Аугментация включена: {augmentation_type}")
    else:
        st.info("🎯 Аугментация отключена")
    
    # Сравнение оригинальных и улучшенных изображений
    if st.session_state.enhancement_applied:
        st.subheader("Сравнение: оригинал vs улучшенное")
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            # Оригинальное изображение
            axes[0, i].imshow(x_train[i])
            axes[0, i].set_title(f"Оригинал: {class_names[y_train[i][0]]}")
            axes[0, i].axis('off')
            
            # Улучшенное изображение
            axes[1, i].imshow(st.session_state.x_train_enhanced[i])
            axes[1, i].set_title(f"Улучшенное: {class_names[y_train[i][0]]}")
            axes[1, i].axis('off')
        st.pyplot(fig)
    
    # Покажем примеры изображений
    st.subheader("Примеры изображений из датасета")
    
    # Определяем максимальный индекс для безопасного доступа
    max_safe_index = min(sample_size, len(x_train_small))
    
    if 'example_indices' not in st.session_state:
        st.session_state.example_indices = []
        for class_id in range(10):
            # Ищем индексы только в пределах безопасного диапазона
            class_indices = np.where(y_train[:max_safe_index] == class_id)[0]
            if len(class_indices) > 0:
                st.session_state.example_indices.append(np.random.choice(class_indices))
            else:
                # Если нет изображений этого класса в выборке, берем из полного датасета
                class_indices_full = np.where(y_train == class_id)[0]
                if len(class_indices_full) > 0:
                    safe_index = min(np.random.choice(class_indices_full), max_safe_index - 1)
                    st.session_state.example_indices.append(safe_index)

    if st.button('🔄 Обновить примеры'):
        st.session_state.example_indices = []
        for class_id in range(10):
            # Ищем индексы только в пределах безопасного диапазона
            class_indices = np.where(y_train[:max_safe_index] == class_id)[0]
            if len(class_indices) > 0:
                st.session_state.example_indices.append(np.random.choice(class_indices))
            else:
                # Если нет изображений этого класса в выборке, берем из полного датасета
                class_indices_full = np.where(y_train == class_id)[0]
                if len(class_indices_full) > 0:
                    safe_index = min(np.random.choice(class_indices_full), max_safe_index - 1)
                    st.session_state.example_indices.append(safe_index)

    # Отрисовка изображений
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, idx in enumerate(st.session_state.example_indices):
        ax = axes[i//5, i%5]
        
        # Безопасный доступ к изображениям
        if idx < len(x_train_small):
            # Используем текущие данные для обучения (x_train_small)
            ax.imshow(x_train_small[idx])
            ax.set_title(f"{class_names[y_train_small[idx]]}")
        else:
            # Если индекс выходит за границы, показываем сообщение
            ax.text(0.5, 0.5, f"Индекс {idx}\nвне диапазона", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title("Ошибка доступа")
        
        ax.axis('off')
    
    st.pyplot(fig)
    
    # Информация о распределении классов
    st.subheader("Распределение классов")
    train_counts = [np.sum(y_train_small == i) for i in range(10)]
    test_counts = [np.sum(y_test_small == i) for i in range(10)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(class_names, train_counts)
    ax1.set_title("Распределение в тренировочной выборке")
    ax1.tick_params(axis='x', rotation=45)
    ax2.bar(class_names, test_counts)
    ax2.set_title("Распределение в тестовой выборке")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Раздел 2: K-Nearest Neighbors (простая модель)
elif section == "K-Nearest Neighbors":
    st.header("🧮 K-Nearest Neighbors (K-NN)")
    st.write("""
    Начнём с простой модели! K-NN ищет наиболее похожие изображения в тренировочной выборке
    и предсказывает класс на основе "соседей".
    """)
    
    st.info("ℹ️ Для K-NN аугментация данных не применяется - модель работает с исходными признаками")

    # Поле для имени эксперимента
    experiment_timestamp = int(time.time())
    experiment_name = st.text_input("Название эксперимента", f"KNN_Experiment_{experiment_timestamp}")
    
    # Автоматический подбор оптимального K
    st.subheader("Автоматический подбор параметров")
    
    if st.button("Найти оптимальное K"):
        x_train_flat = x_train_small.reshape(x_train_small.shape[0], -1)
        optimal_k, elbow_plot = find_optimal_k(x_train_flat, y_train_small)
        st.session_state.optimal_k = optimal_k
        st.pyplot(elbow_plot)
        st.success(f"Рекомендованное значение K: {optimal_k}")

    if 'knn_initialized' not in st.session_state:
        st.session_state.knn_initialized = True
    
    default_k = st.session_state.get('optimal_k', 5)
    k_value = st.slider("Выберите количество соседей (K)", 1, 15, default_k, key="knn_unique_slider")    

    if st.button("Обучить K-NN модель"):
        with st.spinner('Обучаем K-NN... это может занять несколько минут'):
            start_time = time.time()    

            # Подготовка данных для K-NN (выравниваем изображения в векторы)
            x_train_flat = x_train_small.reshape(x_train_small.shape[0], -1)
            x_test_flat = x_test_small.reshape(x_test_small.shape[0], -1)
            
            # Обучение модели
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(x_train_flat, y_train_small)
       
            # Предсказание на тестовой выборке
            y_pred = knn.predict(x_test_flat)
            y_true = y_test_small
            
            end_time = time.time()
            
            # Расширенные метрики
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            st.success(f"K-NN обучен за {end_time - start_time:.2f} секунд!")
            
            # Показываем метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            
            # Сохраняем модель
            model_filename = f"models/knn_model_{experiment_timestamp}.pkl"
            result = save_model_with_check(knn, model_filename, 'knn', experiment_name)
            if result is None:
                save_success = False
                save_message = "Функция сохранения вернула None"
            else:
                save_success, save_message = result
            
            model_size_mb = get_model_size(knn, 'knn')
            
            if save_success:
                st.success(f"✅ {save_message}")
            else:
                st.warning(f"⚠️ {save_message}")
                model_filename = "not_saved"
            
            # Подготавливаем данные для сохранения
            experiment_data = {
                'experiment_name': experiment_name,
                'model_type': 'K-NN',
                'model_filename': model_filename,
                'parameters': json.dumps({
                    'k_value': k_value,
                    'algorithm': 'auto',
                    'weights': 'uniform'
                }),
                'dataset_info': json.dumps({
                    'sample_size': sample_size,
                    'enhancement_applied': st.session_state.enhancement_applied,
                    'augmentation_type': augmentation_type,
                    'data_type': current_data_type
                }),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'enhancement_applied': st.session_state.enhancement_applied,
                'augmentation_type': augmentation_type,
                'sample_size': sample_size,
                'model_size_mb': model_size_mb,
                'save_status': 'success' if save_success else 'failed'
            }
            
            # Сохраняем в базу данных
            save_experiment_to_db(experiment_data)
            st.success(f"Эксперимент '{experiment_name}' сохранен в базу данных!")
            
            # Сохраняем модель для анализа
            st.session_state.knn_model = knn
            st.session_state.knn_accuracy = accuracy
            
            # Покажем несколько примеров предсказаний
            st.subheader("Примеры предсказаний")
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(min(10, len(y_true))):
                ax = axes[i//5, i%5]
                ax.imshow(x_test_small[i])  # Используем текущие тестовые данные
                true_label = class_names[y_true[i]]
                pred_label = class_names[y_pred[i]]
                color = 'green' if y_true[i] == y_pred[i] else 'red'
                ax.set_title(f"Истино: {true_label}\nПредсказано: {pred_label}", color=color)
                ax.axis('off')
            st.pyplot(fig)

# Раздел 3: Простая нейронная сеть
elif section == "Нейронная сеть":
    st.header("🧠 Полносвязная нейронная сеть")
    
    # Поле для имени эксперимента
    experiment_timestamp = int(time.time())
    experiment_name = st.text_input("Название эксперимента", f"NN_Experiment_{experiment_timestamp}")
    
    # Автоматическая рекомендация параметров
    recommended_params = recommend_nn_parameters(sample_size)
    
    st.subheader("Рекомендованные параметры")
    st.write(f"Размер выборки: {sample_size} → Эпохи: {recommended_params['epochs']}, "
             f"Батч: {recommended_params['batch_size']}, "
             f"Нейроны: {recommended_params['units1']}/{recommended_params['units2']}")
    
    epochs = st.slider("Количество эпох", 1, 30, recommended_params['epochs'])
    batch_size = st.slider("Размер батча", 32, 256, recommended_params['batch_size'])
    units1 = st.slider("Нейроны в первом слое", 32, 512, recommended_params['units1'])
    units2 = st.slider("Нейроны во втором слое", 16, 256, recommended_params['units2'])
    dropout_rate = st.slider("Dropout rate", 0.1, 0.7, recommended_params['dropout'])
    
    if st.button("Обучить нейронную сеть"):
        with st.spinner('Обучаем нейронную сеть...'):
            start_time = time.time()
            
            # Подготовка данных
            x_train_flat = x_train_small.reshape(x_train_small.shape[0], -1)
            x_test_flat = x_test_small.reshape(x_test_small.shape[0], -1)
            
            y_train_categorical = to_categorical(y_train_small, 10)
            y_test_categorical = to_categorical(y_test_small, 10)
            
            # Создаём модель
            model = Sequential([
                Dense(units1, activation='relu', input_shape=(3072,)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(units2, activation='relu'),
                BatchNormalization(),
                Dropout(dropout_rate * 0.8),
                Dense(10, activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            
            # Обучение с аугментацией из настроек
            aug_params = get_augmentation_parameters(augmentation_type)
            
            if enable_augmentation and aug_params is not None:
                # Для полносвязной сети нужно выровненные данные
                # Переформатируем данные для аугментации и обратно
                x_train_reshaped = x_train_small.reshape(x_train_small.shape[0], 32, 32, 3)
                
                datagen = ImageDataGenerator(**aug_params)
                
                # Обучаем с аугментацией
                history = model.fit(
                    datagen.flow(x_train_reshaped, y_train_categorical, batch_size=batch_size),
                    steps_per_epoch=len(x_train_reshaped) // batch_size,
                    epochs=epochs,
                    validation_data=(x_test_flat, y_test_categorical),
                    verbose=0
                )
                st.info(f"🎯 Обучение с аугментацией: {augmentation_type}")
            else:
                # Обучаем без аугментации
                history = model.fit(
                    x_train_flat, y_train_categorical,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test_flat, y_test_categorical),
                    verbose=0
                )
            
            # Оценка модели
            test_loss, test_accuracy = model.evaluate(x_test_flat, y_test_categorical, verbose=0)
            
            # Предсказания для метрик
            y_pred_proba = model.predict(x_test_flat, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y_test_small
            
            # Расширенные метрики
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            end_time = time.time()
            
            st.success(f"Нейронная сеть обучена за {end_time - start_time:.2f} секунд!")
            
            # Показываем метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            
            # Сохраняем модель и эксперимент
            model_filename = f"models/nn_model_{experiment_timestamp}.h5"
            save_success, save_message = save_model_with_check(
                model, model_filename, 'neural_network', experiment_name
            )
            
            model_size_mb = get_model_size(model, 'neural_network')
            
            if save_success:
                st.success(f"✅ {save_message}")
            else:
                st.warning(f"⚠️ {save_message}")
                model_filename = "not_saved"
            
            # Подготавливаем данные для сохранения
            experiment_data = {
                'experiment_name': experiment_name,
                'model_type': 'Neural Network',
                'model_filename': model_filename,
                'parameters': json.dumps({
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'units1': units1,
                    'units2': units2,
                    'dropout_rate': dropout_rate,
                    'optimizer': 'adam',
                    'augmentation': augmentation_type
                }),
                'dataset_info': json.dumps({
                    'sample_size': sample_size,
                    'enhancement_applied': st.session_state.enhancement_applied,
                    'augmentation_type': augmentation_type,
                    'data_type': current_data_type
                }),
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'enhancement_applied': st.session_state.enhancement_applied,
                'augmentation_type': augmentation_type,
                'sample_size': sample_size,
                'model_size_mb': model_size_mb,
                'save_status': 'success' if save_success else 'failed'
            }
            
            # Сохраняем в базу данных
            save_experiment_to_db(experiment_data)
            st.success(f"Эксперимент '{experiment_name}' сохранен в базу данных!")
            
            # Сохраняем модель для анализа
            st.session_state.nn_model = model
            st.session_state.nn_history = history
            
            # График обучения
            st.subheader("График обучения")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'], label='Точность на обучении')
            ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
            ax1.set_title('Точность модели')
            ax1.set_xlabel('Эпоха')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Потери на обучении')
            ax2.plot(history.history['val_loss'], label='Потери на валидации')
            ax2.set_title('Потери модели')
            ax2.set_xlabel('Эпоха')
            ax2.legend()
            
            st.pyplot(fig)

# Раздел 4: Свёрточная нейронная сеть (CNN)
elif section == "Свёрточная сеть (CNN)":
    st.header("🎯 Свёрточная нейронная сеть (CNN)")
    
    # Поле для имени эксперимента
    experiment_timestamp = int(time.time()) 
    experiment_name = st.text_input("Название эксперимента", f"CNN_Experiment_{experiment_timestamp}")
    
    # Автоматическая рекомендация параметров
    recommended_params = recommend_cnn_parameters(sample_size)
    
    st.subheader("Рекомендованные параметры")
    st.write(f"Размер выборки: {sample_size} → Эпохи: {recommended_params['epochs']}, "
             f"Батч: {recommended_params['batch_size']}, "
             f"Фильтры: {recommended_params['filters1']}/{recommended_params['filters2']}")
    
    cnn_epochs = st.slider("Количество эпох", 1, 30, recommended_params['epochs'])
    cnn_batch_size = st.slider("Размер батча", 32, 256, recommended_params['batch_size'])
    filters1 = st.slider("Фильтры в первом слое", 16, 128, recommended_params['filters1'])
    filters2 = st.slider("Фильтры во втором слое", 32, 256, recommended_params['filters2'])
    dense_units = st.slider("Нейроны в полносвязном слое", 32, 512, recommended_params['dense_units'])
    
    if st.button("Обучить CNN"):
        with st.spinner('Обучаем свёрточную сеть... это займёт некоторое время'):
            start_time = time.time()
            
            # Подготовка данных
            y_train_categorical = to_categorical(y_train_small, 10)
            y_test_categorical = to_categorical(y_test_small, 10)
            
            # Создаём CNN модель
            model = Sequential([
                Conv2D(filters1, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(filters2, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Conv2D(filters2, (3, 3), activation='relu'),
                BatchNormalization(),
                Flatten(),
                Dense(dense_units, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            
            # ОБУЧЕНИЕ С АУГМЕНТАЦИЕЙ ИЗ НАСТРОЕК
            aug_params = get_augmentation_parameters(augmentation_type)
            
            if enable_augmentation and aug_params is not None:
                datagen = ImageDataGenerator(**aug_params)
                
                history = model.fit(
                    datagen.flow(x_train_small, y_train_categorical, batch_size=cnn_batch_size),
                    steps_per_epoch=len(x_train_small) // cnn_batch_size,
                    epochs=cnn_epochs,
                    validation_data=(x_test_small, y_test_categorical),
                    verbose=0
                )
                st.info(f"🎯 Обучение с аугментацией: {augmentation_type}")
            else:
                history = model.fit(
                    x_train_small, y_train_categorical,
                    epochs=cnn_epochs, batch_size=cnn_batch_size,
                    validation_data=(x_test_small, y_test_categorical),
                    verbose=0
                )
            
            # Оценка модели
            test_loss, test_accuracy = model.evaluate(x_test_small, y_test_categorical, verbose=0)
            
            # Предсказания для метрик
            y_pred_proba = model.predict(x_test_small, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y_test_small
            
            # Расширенные метрики
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            end_time = time.time()
            
            st.success(f"CNN обучена за {end_time - start_time:.2f} секунд!")
            
            # Показываем метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            
            # Сохраняем модель и эксперимент
            model_filename = f"models/cnn_model_{experiment_name}.h5"
            save_success, save_message = save_model_with_check(
                model, model_filename, 'cnn', experiment_name
            )
            
            model_size_mb = get_model_size(model, 'cnn')
            
            if save_success:
                st.success(f"✅ {save_message}")
            else:
                st.warning(f"⚠️ {save_message}")
                model_filename = "not_saved"
            
            # Подготавливаем данные для сохранения
            experiment_data = {
                'experiment_name': experiment_name,
                'model_type': 'CNN',
                'model_filename': model_filename,
                'parameters': json.dumps({
                    'epochs': cnn_epochs,
                    'batch_size': cnn_batch_size,
                    'filters1': filters1,
                    'filters2': filters2,
                    'dense_units': dense_units,
                    'optimizer': 'adam',
                    'augmentation': augmentation_type
                }),
                'dataset_info': json.dumps({
                    'sample_size': sample_size,
                    'enhancement_applied': st.session_state.enhancement_applied,
                    'augmentation_type': augmentation_type,
                    'data_type': current_data_type
                }),
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'enhancement_applied': st.session_state.enhancement_applied,
                'augmentation_type': augmentation_type,
                'sample_size': sample_size,
                'model_size_mb': model_size_mb,
                'save_status': 'success' if save_success else 'failed'
            }
            
            # Сохраняем в базу данных
            save_experiment_to_db(experiment_data)
            st.success(f"Эксперимент '{experiment_name}' сохранен в базу данных!")
            
            # Сохраняем модель для анализа
            st.session_state.cnn_model = model
            st.session_state.cnn_history = history
            
            # График обучения
            st.subheader("График обучения")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'], label='Точность на обучении')
            ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
            ax1.set_title('Точность модели')
            ax1.set_xlabel('Эпоха')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Потери на обучении')
            ax2.plot(history.history['val_loss'], label='Потери на валидации')
            ax2.set_title('Потери модели')
            ax2.set_xlabel('Эпоха')
            ax2.legend()
            
            st.pyplot(fig)

# Раздел 5: Анализ результатов
elif section == "Анализ результатов":
    st.header("📈 Анализ результатов")
    
    # Проверяем, какие модели были обучены
    trained_models = []
    if 'knn_model' in st.session_state:
        trained_models.append('K-NN')
    if 'nn_model' in st.session_state:
        trained_models.append('Neural Network')
    if 'cnn_model' in st.session_state:
        trained_models.append('CNN')
    
    if not trained_models:
        st.warning("Сначала обучите хотя бы одну модель в соответствующих разделах.")
        st.info("""
        **Доступные модели для обучения:**
        - K-Nearest Neighbors (K-NN) - быстрая и простая модель
        - Neural Network - полносвязная нейронная сеть  
        - CNN - свёрточная нейронная сеть (лучшая для изображений)
        """)
    else:
        # Выбор модели для анализа
        selected_model = st.selectbox("Выберите модель для анализа:", trained_models)
        
        if selected_model == 'K-NN':
            # Анализ для K-NN
            knn = st.session_state.knn_model
            x_test_flat = x_test_small.reshape(x_test_small.shape[0], -1)
            y_pred = knn.predict(x_test_flat)
            y_true = y_test_small
            y_pred_proba = None
            
            # Показываем точность K-NN
            accuracy = st.session_state.get('knn_accuracy', accuracy_score(y_true, y_pred))
            st.metric("Точность модели", f"{accuracy:.2%}")
            
        elif selected_model == 'Neural Network':
            # Анализ для нейросети
            model = st.session_state.nn_model
            history = st.session_state.nn_history
            x_test_flat = x_test_small.reshape(x_test_small.shape[0], -1)
            y_pred_proba = model.predict(x_test_flat, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y_test_small
            
            # Показываем график обучения
            st.subheader("График обучения")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'], label='Точность на обучении')
            ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
            ax1.set_title('Точность модели')
            ax1.set_xlabel('Эпоха')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Потери на обучении')
            ax2.plot(history.history['val_loss'], label='Потери на валидации')
            ax2.set_title('Потери модели')
            ax2.set_xlabel('Эпоха')
            ax2.legend()
            st.pyplot(fig)
            
        elif selected_model == 'CNN':
            # Анализ для CNN
            model = st.session_state.cnn_model
            history = st.session_state.cnn_history
            y_pred_proba = model.predict(x_test_small, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y_test_small
            
            # Показываем график обучения
            st.subheader("График обучения")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.plot(history.history['accuracy'], label='Точность на обучении')
            ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
            ax1.set_title('Точность модели')
            ax1.set_xlabel('Эпоха')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Потери на обучении')
            ax2.plot(history.history['val_loss'], label='Потери на валидации')
            ax2.set_title('Потери модели')
            ax2.set_xlabel('Эпоха')
            ax2.legend()
            st.pyplot(fig)
        
        # Общий анализ для всех моделей
        st.subheader(f"Детальный анализ для модели: {selected_model}")
        
        # Создаем расширенные графики
        plots, metrics_df = create_extended_plots(y_true, y_pred, y_pred_proba, class_names)
        
        # Показываем графики
        for plot_name, plot in plots.items():
            st.subheader(plot_name.replace('_', ' ').title())
            st.pyplot(plot)
        
        # Отчёт по классификации
        st.subheader("Отчёт по классификации")
        st.dataframe(metrics_df)
        
        # Анализ ошибок
        st.subheader("Примеры ошибок классификации")
        error_indices = np.where(y_pred != y_true)[0]
        
        if len(error_indices) > 0:
            show_errors = min(10, len(error_indices))
            error_samples = np.random.choice(error_indices, show_errors, replace=False)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.ravel()
            
            for i, idx in enumerate(error_samples):
                if idx < len(x_test_small):  # Безопасный доступ
                    axes[i].imshow(x_test_small[idx])
                    true_label = class_names[y_true[idx]]
                    pred_label = class_names[y_pred[idx]]
                    
                    # Для моделей с вероятностями показываем уверенность
                    confidence = None
                    if selected_model in ['Neural Network', 'CNN'] and y_pred_proba is not None:
                        confidence = np.max(y_pred_proba[idx])
                    
                    title = f"Истино: {true_label}\nПредсказано: {pred_label}"
                    if confidence is not None:
                        title += f"\nУверенность: {confidence:.2f}"
                    
                    axes[i].set_title(title, color='red')
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, "Ошибка\nдоступа", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            # Скрываем пустые subplots
            for i in range(show_errors, 10):
                axes[i].set_visible(False)
                
            st.pyplot(fig)
        else:
            st.success("Отличный результат! Нет ошибок классификации на тестовой выборке.")

# Раздел 6: Сравнение экспериментов
elif section == "Сравнение экспериментов":
    st.header("📊 Сравнение всех экспериментов")
    
    # Загружаем эксперименты из базы данных
    experiments_df = load_experiments_from_db()
    
    if experiments_df.empty:
        st.warning("Пока нет проведённых экспериментов. Обучите хотя бы одну модель.")
    else:
        st.subheader("Все сохраненные эксперименты")
        
        # Показываем полную таблицу с улучшенным форматированием
        display_df = experiments_df.copy()
        display_df['created_date'] = pd.to_datetime(display_df['created_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Форматируем числовые колонки
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'model_size_mb']
        for col in numeric_cols:
            if col in display_df.columns:
                if col == 'training_time':
                    display_df[col] = display_df[col].round(2)
                elif col == 'model_size_mb':
                    display_df[col] = display_df[col].round(1)
                else:
                    display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
        
        st.dataframe(display_df)
        
        # Визуализация сравнения
        st.subheader("Визуальное сравнение моделей")
        
        # 1. Сравнение точности по моделям
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        models = experiments_df['model_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_data = experiments_df[experiments_df['model_type'] == model]
            ax1.scatter(model_data['created_date'], model_data['accuracy'], 
                       label=model, color=colors[i], s=100, alpha=0.7)
        
        ax1.set_xlabel('Дата эксперимента')
        ax1.set_ylabel('Точность')
        ax1.set_title('Динамика точности по времени')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
        
        # 2. Сравнение метрик
        st.subheader("Сравнение метрик по моделям")
        
        metrics_comparison = experiments_df.groupby('model_type').agg({
            'accuracy': ['mean', 'std', 'max'],
            'precision': 'mean',
            'recall': 'mean', 
            'f1_score': 'mean',
            'training_time': 'mean',
            'model_size_mb': 'mean'
        }).round(4)

        metrics_comparison.columns = [
            'accuracy_mean', 'accuracy_std', 'accuracy_max',
            'precision_mean', 'recall_mean', 'f1_mean',
            'training_time_mean', 'model_size_mb_mean'
        ]

        metrics_comparison = metrics_comparison.fillna(0)

        st.dataframe(metrics_comparison)
        
        # 3. Влияние улучшения качества
        st.subheader("Анализ влияния улучшения качества")
        
        enhanced_data = experiments_df[experiments_df['enhancement_applied'] == True]
        original_data = experiments_df[experiments_df['enhancement_applied'] == False]
        
        if not enhanced_data.empty and not original_data.empty:
            fig2, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Точность с улучшением и без
            categories = ['С улучшением', 'Без улучшения']
            accuracy_means = [enhanced_data['accuracy'].mean(), original_data['accuracy'].mean()]
            accuracy_stds = [enhanced_data['accuracy'].std(), original_data['accuracy'].std()]
            
            bars1 = axes[0].bar(categories, accuracy_means, yerr=accuracy_stds, capsize=5, alpha=0.7, color=['lightgreen', 'lightcoral'])
            axes[0].set_ylabel('Точность')
            axes[0].set_title('Влияние улучшения качества на точность')
            axes[0].grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars1, accuracy_means):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Время обучения
            time_means = [enhanced_data['training_time'].mean(), original_data['training_time'].mean()]
            time_stds = [enhanced_data['training_time'].std(), original_data['training_time'].std()]
            
            bars2 = axes[1].bar(categories, time_means, yerr=time_stds, capsize=5, alpha=0.7, color=['lightblue', 'orange'])
            axes[1].set_ylabel('Время обучения (сек)')
            axes[1].set_title('Влияние улучшения качества на время обучения')
            axes[1].grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars2, time_means):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}с', ha='center', va='bottom')
            
            st.pyplot(fig2)
        
        # 4. Размеры моделей
        st.subheader("Размеры сохраненных моделей")
        
        if 'model_size_mb' in experiments_df.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            saved_models = experiments_df[experiments_df['save_status'] == 'success']
            
            if not saved_models.empty:
                model_sizes = saved_models.groupby('model_type')['model_size_mb'].mean()
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                bars = ax3.bar(model_sizes.index, model_sizes.values, color=colors[:len(model_sizes)], alpha=0.7)
                
                ax3.set_ylabel('Размер модели (MB)')
                ax3.set_title('Средний размер моделей по типам')
                ax3.grid(True, alpha=0.3)
                
                # Добавляем значения на столбцы
                for bar, value in zip(bars, model_sizes.values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.1f}MB', ha='center', va='bottom')
                
                st.pyplot(fig3)
        
        # 5. Лучшие модели по каждому типу
        st.subheader("Лучшие модели по каждому типу")
        best_models = experiments_df.loc[experiments_df.groupby('model_type')['accuracy'].idxmax()]
        best_display = best_models[['experiment_name', 'model_type', 'accuracy', 'precision', 'recall', 'f1_score', 'created_date', 'model_size_mb']].copy()
        best_display['accuracy'] = (best_display['accuracy'] * 100).round(2).astype(str) + '%'
        best_display['created_date'] = pd.to_datetime(best_display['created_date']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(best_display)
        
        # 6. Детальный анализ выбранного эксперимента
        st.subheader("Детальный анализ эксперимента")
        selected_experiment = st.selectbox("Выберите эксперимент для детального анализа:", 
                                        experiments_df['experiment_name'].values)

        if selected_experiment:
            exp_data = experiments_df[experiments_df['experiment_name'] == selected_experiment].iloc[0]
            
            # Метрики производительности
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{exp_data['accuracy']:.2%}")
                st.metric("Precision", f"{exp_data['precision']:.2%}")
            with col2:
                st.metric("Recall", f"{exp_data['recall']:.2%}")
                st.metric("F1-Score", f"{exp_data['f1_score']:.2%}")
            with col3:
                st.metric("Время обучения", f"{exp_data['training_time']:.1f}с")
                if exp_data['model_size_mb']:
                    st.metric("Размер модели", f"{exp_data['model_size_mb']:.1f}MB")
            
            # Информация о параметрах
            st.write("**⚙️ Параметры модели:**")
            st.json(json.loads(exp_data['parameters']))
            
            st.write("**📁 Информация о данных:**")
            st.json(json.loads(exp_data['dataset_info']))
            
            # Информация о модели
            st.subheader("💾 Информация о модели")
            
            if exp_data['save_status'] == 'success' and exp_data['model_filename'] != 'not_saved':
                st.success(f"✅ Модель сохранена в файле: `{exp_data['model_filename']}`")
                
                # Показываем размер модели если есть
                if exp_data['model_size_mb'] and exp_data['model_size_mb'] > 0:
                    st.info(f"📦 Размер модели: **{exp_data['model_size_mb']:.1f} MB**")
                
                # Информация о доступности файла
                if os.path.exists(exp_data['model_filename']):
                    file_size = os.path.getsize(exp_data['model_filename']) / (1024 * 1024)
                    st.success(f"📁 Файл модели доступен на диске ({file_size:.1f} MB)")
                else:
                    st.warning("⚠️ Файл модели не найден на диске")
                    
            else:
                st.warning("⚠️ Модель не была сохранена (отключено в настройках или превышен лимит размера)")
            
            # Дополнительная информация об эксперименте
            st.info(f"""
            **📊 Детали эксперимента:**
            - **Дата создания**: {exp_data['created_date']}
            - **Размер выборки**: {exp_data['sample_size']} изображений
            - **Улучшение качества**: {'✅ Применено' if exp_data['enhancement_applied'] else '❌ Не применено'}
            - **Аугментация**: {exp_data['augmentation_type']}
            - **Время обучения**: {exp_data['training_time']:.1f} секунд
            - **Статус сохранения**: {'✅ Успешно' if exp_data['save_status'] == 'success' else '❌ Не сохранено'}
            """)
    
    df = export_experiments_to_csv()
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать данные экспериментов (CSV)",
        data=csv,
        file_name="experiments_data.csv",
        mime="text/csv"
    )        

# Информация о проекте в боковой панели
st.sidebar.markdown("---")
st.sidebar.subheader("Статус системы")

# Проверка структуры проекта
project_status = check_project_structure()
for folder, info in project_status.items():
    if info['exists'] and info['is_directory']:
        if info['file_count'] > 0:
            st.sidebar.success(f"📁 {folder}/ ({info['file_count']} файлов)")
        else:
            st.sidebar.info(f"📁 {folder}/ (пустая)")
    else:
        st.sidebar.error(f"❌ {folder}/ (отсутствует)")

# Статус базы данных
st.sidebar.write("**База данных:**")
if db_existed:
    st.sidebar.info(f"📊 Существующая ({initial_records} записей)")
else:
    st.sidebar.success("📊 Новая (создана)")

# Статус сохранения моделей
st.sidebar.write("**Сохранение моделей:**")
for model_type, settings in SAVE_SETTINGS.items():
    status = "✅" if settings['enabled'] else "❌"
    st.sidebar.write(f"{status} {model_type}")

st.sidebar.markdown("---")
st.sidebar.subheader("Статус улучшения")
if st.session_state.enhancement_applied:
    st.sidebar.success("✅ Улучшение применено")
else:
    st.sidebar.info("ℹ️ Оригинальные изображения")

st.sidebar.subheader("Статус аугментации")
if enable_augmentation:
    st.sidebar.success(f"✅ Аугментация: {augmentation_type}")
else:
    st.sidebar.info("ℹ️ Аугментация отключена")