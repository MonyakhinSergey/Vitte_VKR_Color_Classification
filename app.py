# -*- coding: utf-8 -*-
"""
app.py — Интеллектуальный кредитный калькулятор
"""

# ── автозапуск через stcli / runtime ─────────────────────────
from streamlit.web import cli as stcli
import sys
from streamlit import runtime

# ── стандартные импорты ──────────────────────────────────────
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from math import pow, ceil
from catboost import CatBoostClassifier, Pool, CatBoostError
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── увеличиваем лимит для Pandas Styler, чтобы окрашивать строки ──
pd.set_option("styler.render.max_elements", 10**7)
