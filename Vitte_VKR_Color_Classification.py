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

import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути к основным директориям
base_dir = 'datasets/combined_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Удаляем старые директории, если они существуют, и создаем новые
for directory in [train_dir, val_dir, test_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Получаем только папки с цветами, исключая скрытые системные файлы
color_classes = [folder for folder in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, folder)) and not folder.startswith('.')]

# Перераспределение данных: 70% - train, 20% - val, 10% - test
for color in color_classes:
    class_path = os.path.join(base_dir, color)
    images = [img for img in os.listdir(class_path) if not img.startswith('.') and os.path.isfile(os.path.join(class_path, img))]
    random.shuffle(images)

    # Вычисляем количество данных для каждой выборки
    num_images = len(images)
    num_train = int(num_images * 0.7)
    num_val = int(num_images * 0.2)

    # Разделяем данные
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Создаем директории для текущего класса
    os.makedirs(os.path.join(train_dir, color), exist_ok=True)
    os.makedirs(os.path.join(val_dir, color), exist_ok=True)
    os.makedirs(os.path.join(test_dir, color), exist_ok=True)

    # Перемещаем изображения
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, color, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, color, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, color, img))

print("Данные успешно распределены по выборкам.")

# Параметры изображений и размер батча
IMG_HEIGHT = 227
IMG_WIDTH = 227
BATCH_SIZE = 32

# Создаем генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Генератор для обучающих данных
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# Генератор для валидационных данных
val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# Генератор для тестовых данных
test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Печать классов для проверки
print("Классы в обучающих данных:", train_generator.class_indices)
print("Классы в тестовых данных:", test_generator.class_indices)

# Словарь классов
class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Чтобы визуализировать пару изображений "до" и "после", можно получить batch из train_generator
x_batch, y_batch = next(train_generator)  # Получаем пакет (X, Y)
# x_batch у нас уже в виде float32 [0..1], но можно предположить "до" аугментации -> показать ещё raw

# Выведем первые 4 изображения
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(x_batch[i])
    axes[i].set_title("Sample (augmented)")
    axes[i].axis('off')
plt.suptitle("Примеры изображений после предобработки/аугментации")
plt.show()

# 3. Базовая (Baseline) модель

# В этом блоке мы определим нашу базовую модель (Baseline): простую CNN без всяких ухищрений, чтобы потом сравнить её с остальными. После определения модели выведем её архитектуру и сохраним схему.

num_classes = len(class_indices)  # Количество классов (цветов)

# 1. Базовая модель
def create_baseline_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

baseline_model = create_baseline_model()
baseline_model.summary()
plot_model(baseline_model, to_file='baseline_model.png', show_shapes=True)

# Предобученная (Pretrained) модель MobileNetV2

# В этом блоке мы определим вторую модель (Pretrained), например, на основе MobileNetV2. Включим глобальный pooling и финальные Dense-слои. Входные изображения нужно пропускать через функцию mobilenet_preprocess, поэтому создадим ещё один генератор или добавим шаг препроцессинга.

base_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_mobilenet.trainable = False  # Замораживаем базовые слои

def create_pretrained_model(base_model, classes=num_classes):
    # Входная модель (предобученная)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)  # Исправлено: корректный вызов BatchNormalization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)  # Исправлено
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(classes, activation='softmax')(x)

    # Создаём модель
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

pretrained_model = create_pretrained_model(base_mobilenet)
pretrained_model.summary()
plot_model(pretrained_model, to_file='pretrained_model.png', show_shapes=True)

# Модифицированная (Modified) модель

# В этом блоке мы определим модифицированную модель (Modified). Допустим, мы добавим skip-коннекты или, например, attention-модуль или двуглавую структуру. Для наглядности сделаем нечто чуть более сложное, чем Baseline: добавим боковую ветвь и объединим. Это очень условный пример, но демонстрирует идею.

def create_modified_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=num_classes):
    inputs = layers.Input(shape=input_shape)

    # Основная ветвь A
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Боковая ветвь B (Skip Connection)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D((2, 2))(y)
    y = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.GlobalAveragePooling2D()(y)

    # Проверяем размерности ветвей для объединения
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Убедимся, что размеры совместимы перед объединением
    x_shape = tf.keras.backend.int_shape(x)
    y_shape = tf.keras.backend.int_shape(y)
    if len(x_shape) != len(y_shape):
        y = layers.Reshape((1,))(y)  # Приведение y к совместимой форме

    concat = layers.concatenate([x, y])

    # Несколько Dense-слоёв после concat
    fc = layers.Dense(256, activation='relu')(concat)
    fc = layers.BatchNormalization()(fc)
    fc = layers.Dropout(0.5)(fc)
    fc = layers.Dense(128, activation='relu')(fc)
    fc = layers.Dropout(0.3)(fc)
    outputs = layers.Dense(classes, activation='softmax')(fc)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

modified_model = create_modified_model()
modified_model.summary()

# Альтернативный способ визуализации структуры модели
for i, layer in enumerate(modified_model.layers):
    output_shape = getattr(layer, 'output_shape', 'N/A')  # Проверяем наличие атрибута
    print(f"Layer {i}: {layer.name}, Output Shape: {output_shape}")

!pip install graphviz
from graphviz import Digraph

def visualize_model_architecture(model):
    dot = Digraph(comment='Model Architecture')
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        output_shape = getattr(layer, 'output_shape', 'N/A')
        dot.node(f'Layer_{i}', f"{layer_name}\nOutput Shape: {output_shape}")
        if i > 0:
            dot.edge(f'Layer_{i-1}', f'Layer_{i}')
    dot.render('model_architecture', format='png', cleanup=True)
    return dot

# Визуализация архитектуры
visualize_model_architecture(modified_model)

# Обучение и оценка всех моделей, сбор результатов
EPOCHS = 20

history_dict = {}  # здесь будем хранить истории обучения
results_df = pd.DataFrame(columns=['Model', 'ValAccuracy', 'ValLoss', 'TestAccuracy', 'Precision', 'Recall', 'F1'])

# ========== Baseline ==========
baseline_model = create_baseline_model()
hist_baseline = baseline_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)
history_dict['Baseline'] = hist_baseline

# Оценим на валидации
val_loss_b, val_acc_b = baseline_model.evaluate(val_generator, verbose=0)

# Оценим на тесте
test_loss_b, test_acc_b = baseline_model.evaluate(test_generator, verbose=0)

# Предсказания для precision/recall/f1
y_pred_b = np.argmax(baseline_model.predict(test_generator), axis=1)
y_true_b = test_generator.classes
precision_b = precision_score(y_true_b, y_pred_b, average='macro')
recall_b    = recall_score(y_true_b, y_pred_b, average='macro')
f1_b        = f1_score(y_true_b, y_pred_b, average='macro')

new_row = pd.DataFrame([{
    'Model': 'Baseline',
    'ValAccuracy': val_acc_b,
    'ValLoss': val_loss_b,
    'TestAccuracy': test_acc_b,
    'Precision': precision_b,
    'Recall': recall_b,
    'F1': f1_b
}])

results_df = pd.concat([results_df, new_row], ignore_index=True)

# ========== Pretrained ==========
pretrained_model = create_pretrained_model(base_mobilenet)
hist_pretrained = pretrained_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)
history_dict['Pretrained'] = hist_pretrained

val_loss_p, val_acc_p = pretrained_model.evaluate(val_generator, verbose=0)
test_loss_p, test_acc_p = pretrained_model.evaluate(test_generator, verbose=0)

y_pred_p = np.argmax(pretrained_model.predict(test_generator), axis=1)
y_true_p = test_generator.classes
precision_p = precision_score(y_true_p, y_pred_p, average='macro')
recall_p    = recall_score(y_true_p, y_pred_p, average='macro')
f1_p        = f1_score(y_true_p, y_pred_p, average='macro')

new_row = pd.DataFrame([{
    'Model': 'Pretrained',
    'ValAccuracy': val_acc_p,
    'ValLoss': val_loss_p,
    'TestAccuracy': test_acc_p,
    'Precision': precision_p,
    'Recall': recall_p,
    'F1': f1_p
}])

results_df = pd.concat([results_df, new_row], ignore_index=True)

# ========== Modified ==========
modified_model = create_modified_model()
hist_modified = modified_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)
history_dict['Modified'] = hist_modified

val_loss_m, val_acc_m = modified_model.evaluate(val_generator, verbose=0)
test_loss_m, test_acc_m = modified_model.evaluate(test_generator, verbose=0)

y_pred_m = np.argmax(modified_model.predict(test_generator), axis=1)
y_true_m = test_generator.classes
precision_m = precision_score(y_true_m, y_pred_m, average='macro')
recall_m    = recall_score(y_true_m, y_pred_m, average='macro')
f1_m        = f1_score(y_true_m, y_pred_m, average='macro')

new_row = pd.DataFrame([{
    'Model': 'Modified',
    'ValAccuracy': val_acc_m,
    'ValLoss': val_loss_m,
    'TestAccuracy': test_acc_m,
    'Precision': precision_m,
    'Recall': recall_m,
    'F1': f1_m
}])

results_df = pd.concat([results_df, new_row], ignore_index=True)

print("==== Окончательные результаты ====")
display(results_df)
