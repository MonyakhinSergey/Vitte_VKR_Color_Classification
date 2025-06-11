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
