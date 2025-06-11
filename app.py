import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3
import pandas as pd
from PIL import Image

# Загрузка модели
MODEL_PATH = "modified_model.h5"
model = load_model(MODEL_PATH)

# Классы цветов
class_labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 
                'pink', 'purple', 'red', 'silver', 'white', 'yellow']

# Функция для предсказания класса
def predict_color(image_path):
    img = image.load_img(image_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_labels[class_idx], confidence

# Работа с базой данных
DB_PATH = "predictions.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                predicted_class TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    print("База данных инициализирована.")

def save_prediction(image_path, predicted_class, confidence):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (image_path, predicted_class, confidence)
            VALUES (?, ?, ?)
        """, (image_path, predicted_class, confidence))

def load_predictions():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    return df

# Инициализация базы данных
init_db()

# Streamlit-приложение
def main():
    pass
