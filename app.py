# -*- coding: utf-8 -*-
"""
Streamlit-приложение: определяем доминирующий цвет изображения.
CNN → обязательная проверка GPT-4o-mini (proxyapi.ru).
"""

import os, io, base64, sqlite3, time, numpy as np, pandas as pd
from PIL import Image
import streamlit as st
from functools import lru_cache
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
import openai

# ───────────────────── 1. OpenAI через российский прокси ──────────────────────
PROXY_BASE_URL = "https://api.proxyapi.ru/openai/v1"
PROXY_API_KEY  = "sk-2uHtBOkjr3ZrCn43aUt4WdEZ20JaXu49"   # ← ваш ключ

@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=PROXY_API_KEY,
        base_url=PROXY_BASE_URL,
        timeout=60,
    )

# ────────────────────────── 2. Загрузка CNN-модели ────────────────────────────
MODEL_PATH = "modified_model.h5"
model = load_model(MODEL_PATH)

class_labels = [
    "black", "blue", "brown", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "white", "yellow"
]

def predict_color_cnn(img: Image.Image):
    arr = kimage.img_to_array(img.resize((160, 160))) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    preds = preds[0] if preds.ndim > 1 else preds
    if preds.size == 0:
        return "unknown", 0.0
    idx  = int(np.argmax(preds))
    conf = float(np.max(preds))
    return (class_labels[idx] if idx < len(class_labels) else "unknown", conf)

# ───────────── 3. GPT-4o-mini: подтверждение/исправление результата ───────────
GPT_PROMPT = (
    "You are an expert assistant. Identify the single English word that best "
    "describes the dominant color in the given image. "
    "Only respond with that word in lowercase (e.g. 'red')."
)

def image_to_b64(img: Image.Image, side: int = 256) -> str:
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def correct_color_with_gpt(img: Image.Image, fallback: str) -> str:
    client = get_client()
    b64 = image_to_b64(img)
    messages = [
        {"role": "system", "content": GPT_PROMPT},
        {"role": "user", "content": [
            {"type": "input_text",
             "text": "What is the dominant color in this image? respond only with the color word."},
            {"type": "input_image",
             "image_url": f"data:image/png;base64,{b64}"}
        ]}
    ]
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            temperature=0.0,
            max_output_tokens=16      # ≥ 16   ← исправлено
        )
        if hasattr(resp, "output") and resp.output:
            gpt_ans = resp.output[0].content[0].text.strip().lower()
            if gpt_ans in class_labels:
                return gpt_ans
    except Exception as e:
        st.warning(f"⚠️ GPT-запрос не удался: {e}")
    return fallback

# ─────────────────────────── 4. База данных (SQLite) ──────────────────────────
DB_PATH = "predictions.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # если таблицы нет – создаём со всеми нужными полями
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_path TEXT,
              predicted_class TEXT,
              confidence REAL,
              corrected INTEGER DEFAULT 0,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # если колонка corrected отсутствует (старый вариант таблицы) – добавим
        cur = conn.execute("PRAGMA table_info(predictions)")
        cols = [row[1] for row in cur.fetchall()]
        if "corrected" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN corrected INTEGER DEFAULT 0")

def save_pred(path, cls, conf, corr):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (image_path, predicted_class, confidence, corrected)"
            "VALUES (?,?,?,?)",
            (path, cls, conf, int(corr))
        )

def load_preds():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM predictions", conn)

init_db()

# ─────────────────────────── 5. Streamlit-интерфейс ──────────────────────────
st.set_page_config(page_title="Определение цвета", page_icon="🎨")
st.title("🖼️ Определение доминирующего цвета")

section = st.sidebar.radio("Навигация", ["Загрузить изображение", "Статистика"])

if section == "Загрузить изображение":
    st.subheader("Загрузите изображение")
    uploaded = st.file_uploader("Файл (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Исходное изображение", use_container_width=True)

        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", uploaded.name)
        img.save(path)

        # 1) прогноз CNN
        cnn_class, cnn_conf = predict_color_cnn(img)

        # 2) подтверждение GPT-4o-mini
        final_class = correct_color_with_gpt(img, cnn_class)
        corrected   = final_class != cnn_class

        txt = f"Цвет: **{final_class}**  \nУверенность CNN: {cnn_conf:.2f}"
        if corrected or cnn_class == "unknown":
            txt += "  \n*(уточнено GPT-4o-mini)*"
        st.success(txt)

        save_pred(path, final_class, cnn_conf, corrected)

elif section == "Статистика":
    st.subheader("Статистика предсказаний")
    df = load_preds()
    if df.empty:
        st.info("Пока нет данных.")
    else:
        st.dataframe(df)
        st.bar_chart(df.groupby("predicted_class").size())

# ─────────────────────────── 6. Запуск скрипта ────────────────────────────────
if __name__ == "__main__":
    from streamlit import runtime
    import sys
    from streamlit.web import cli as stcli
    if not runtime.exists():          # запуск из консоли:  python this.py
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
