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
    st.title("Классификация изображений по цвету")
    st.sidebar.title("Навигация")
    menu = st.sidebar.radio("Выберите действие", ["Загрузить изображение", "Просмотреть статистику"])

    if menu == "Загрузить изображение":
        st.header("Загрузите изображение для классификации")
        uploaded_file = st.file_uploader("Выберите файл изображения", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Загруженное изображение", use_column_width=True)

            # Сохранение загруженного файла
            temp_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Предсказание
            predicted_class, confidence = predict_color(temp_path)
            st.success(f"Цвет: {predicted_class} (уверенность: {confidence:.2f})")

            # Сохранение предсказания в БД
            save_prediction(temp_path, predicted_class, confidence)
            st.info("Результат сохранен в базе данных.")

    elif menu == "Просмотреть статистику":
        st.header("Статистика предсказаний")
        data = load_predictions()
        if not data.empty:
            st.dataframe(data)
            st.bar_chart(data.groupby("predicted_class").size())
        else:
            st.info("Статистика пока пуста. Сначала сделайте предсказания.")

# Запуск приложения
if __name__ == '__main__':
    from streamlit import runtime
    import sys
    from streamlit.web import cli as stcli

    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
