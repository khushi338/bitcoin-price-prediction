import streamlit as st
import pandas as pd
import pickle

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide"
)

# ================== DARK CRYPTO THEME ==================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #f7931a;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD TRAINED MODEL ==================
with open("model/btc_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================== LOAD DATA (STREAMLIT CLOUD SAFE) ==================
DATA_URL = "https://stooq.com/q/d/l/?s=btcusd&i=d"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna()

df = load_data()

# ================== MOVING AVERAGES ==================
df["ma7"] = df["close"].rolling(window=7).mean()
df["ma30"] = df["close"].rolling(window=30).mean()

#
