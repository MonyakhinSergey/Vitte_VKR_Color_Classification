# Импорт библиотек и настройка окружения
# В этом блоке мы подключаем нужные библиотеки и проводим базовую настройку окружения. Здесь же можно установить и настроить доступ к Kaggle или Google Drive, если необходимо.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

print("TensorFlow version:", tf.__version__)

# Если нужно, можно настроить kaggle или Google Drive здесь, например:
# from google.colab import drive
# drive.mount('/content/drive')
# или
!pip install kaggle
# и т.д.

# Устанавливаем некоторый параметр для отображения графиков
plt.rcParams['figure.figsize'] = (8, 6)
sns.set_style("whitegrid")

# Загружаем *.json-файл
from google.colab import files
files.upload()

# Создаем директорию для конфигурационного файла
!mkdir -p ~/.kaggle

# Перемещаем загруженный файл kaggle.json в созданную директорию
!cp kaggle.json ~/.kaggle/

# Устанавливаем необходимые права доступа
!chmod 600 ~/.kaggle/kaggle.json

# Проверяем, что Kaggle API настроен корректно
!kaggle datasets list

# Создаем директорию для датасетов
!mkdir -p datasets

# Загружаем первый датасет
!kaggle datasets download -d imoore/6000-store-items-images-classified-by-color -p datasets

# Разархивируем загруженный датасет
!unzip -q datasets/6000-store-items-images-classified-by-color.zip -d datasets/6000_store_items

# Загружаем второй датасет
!kaggle datasets download -d ayanzadeh93/color-classification -p datasets

# Разархивируем загруженный датасет
!unzip -q datasets/color-classification.zip -d datasets/color_classification

'''
[6000+ Store Items Images Classified By Color](https://www.kaggle.com/datasets/imoore/6000-store-items-images-classified-by-color): набор данных содержит более 6000 изображений, представляющих различные товары, классифицированные по их цвету. Каждый элемент датасета распределен по категориям, обозначающим основные цвета, такие как черный, синий, красный и другие. Этот датасет разработан для задач, связанных с визуальной классификацией объектов по цветовым характеристикам, и подходит для исследований в области компьютерного зрения, машинного обучения и анализа изображений. Высокое разнообразие товаров делает его полезным для обучения моделей, работающих с реальными данными в розничной торговле и e-commerce.

[Color Classification:](https://www.kaggle.com/datasets/ayanzadeh93/color-classification) этот датасет ориентирован на решение задачи классификации изображений по цвету. Он содержит изображения, которые распределены по папкам, каждая из которых соответствует определенному цвету, включая основные оттенки, такие как черный, белый, синий и коричневый. Датасет предназначен для разработки и тестирования алгоритмов классификации изображений, особенно в контексте задач сегментации и цветового анализа. Его структура упрощает процесс подготовки данных для обучения моделей и экспериментов, связанных с визуальным восприятием и распознаванием цвета.
'''

# Загрузка данных, предобработка, визуализация

# В этом блоке мы загружаем данные, задаём минимальную предобработку и строим пару изображений для визуализации. Оба датасета уже структурированы по подпапкам, где цвета – это классы.

import os
import shutil

# Пути к исходным папкам
source_1 = 'datasets/6000_store_items/train'
source_2 = 'datasets/color_classification/ColorClassification'
combined_path = 'datasets/combined_dataset'

# Создаем директорию для объединенного набора данных
os.makedirs(combined_path, exist_ok=True)

# Функция для копирования файлов с преобразованием названий цветов в нижний регистр
def copy_and_normalize(source, destination):
    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        if os.path.isdir(folder_path):
            # Приводим название папки к нижнему регистру
            normalized_folder = folder.lower()
            dest_folder = os.path.join(destination, normalized_folder)
            os.makedirs(dest_folder, exist_ok=True)

            # Копируем все файлы из исходной папки
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, dest_folder)

# Копируем данные из обеих исходных папок
copy_and_normalize(source_1, combined_path)
copy_and_normalize(source_2, combined_path)

print("Объединение папок завершено. Все данные приведены к единому регистру.")

import os
import shutil

# Путь к объединенной директории
combined_path = 'datasets/combined_dataset'

# Список допустимых названий цветов
valid_colors = [
    'black', 'blue', 'brown', 'green', 'grey',
    'orange', 'pink', 'purple', 'red', 'silver',
    'white', 'yellow'
]

# Удаляем папки, которые не относятся к цветам
for folder in os.listdir(combined_path):
    folder_path = os.path.join(combined_path, folder)
    if os.path.isdir(folder_path) and folder.lower() not in valid_colors:
        shutil.rmtree(folder_path)
        print(f"Удалена папка: {folder_path}")

print("Формирование датасета завершено. Оставлены только папки с названиями цветов.")
