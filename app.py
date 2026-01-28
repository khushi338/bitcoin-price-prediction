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

# ================== ALWAYS RENDER UI FIRST ==================
st.markdown(
    "<h1 style='text-align:center;'>₿ Bitcoin Price Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning • Technical Indicators • Streamlit</p>",
    unsafe_allow_html=True
)
st.divider()

# ================== LOAD MODEL SAFELY ==================
try:
    with open("model/btc_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Model file not found or corrupted.")
    st.stop()

# ================== LOAD DATA SAFELY ==================
DATA_URL = "https://stooq.com/q/d/l/?s=btcusd&i=d"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna()

with st.spinner("🔄 Loading Bitcoin data..."):
    try:
        df = load_data()
    except Exception as e:
        st.error("❌ Failed to load Bitcoin data from source.")
        st.info("This can happen due to network restrictions on Streamlit Cloud.")
        st.stop()

# ================== BASIC DATA VALIDATION ==================
required_cols = {"open", "high", "low", "close"}
if not required_cols.issubset(df.columns):
    st.error("❌ Dataset schema mismatch.")
    st.write("Columns found:", list(df.columns))
    st.stop()

# ================== MOVING AVERAGES ==================
df["ma7"] = df["close"].rolling(7).mean()
df["ma30"] = df["close"].rolling(30).mean()

# ================== MODEL PREDICTIONS ==================
try:
    X = df[["open", "high", "low"]]
    df["predicted"] = model.predict(X)
except Exception as e:
    st.error("❌ Prediction failed due to feature mismatch.")
    st.stop()

# ================== LIMIT DATA FOR UI ==================
df_ui = df.sort_values("date").tail(730)

# ================== SIDEBAR ==================
st.sidebar.header("📊 Predict Bitcoin Closing Price")

open_price = st.sidebar.number_input("Open Price ($)", value=30000.0)
high_price = st.sidebar.number_input("High Price ($)", value=31000.0)
low_price = st.sidebar.number_input("Low Price ($)", value=29500.0)

if st.sidebar.button("🚀 Predict Price"):
    pred = model.predict([[open_price, high_price, low_price]])
    st.sidebar.success(f"💰 Predicted Close Price: ${pred[0]:,.2f}")

# ================== PRICE + MOVING AVERAGES ==================
st.subheader("📈 Bitcoin Price with MA7 & MA30 (Last 2 Years)")
st.line_chart(
    df_ui.set_index("date")[["close", "ma7", "ma30"]]
)

# ================== ACTUAL vs PREDICTED ==================
st.subheader("📉 Actual vs Predicted Closing Price (Last 2 Years)")
st.line_chart(
    df_ui.set_index("date")[["close", "predicted"]]
)

# ================== FOOTER ==================
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Built with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
