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
