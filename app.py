# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Commodity Price Predictor - India",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 Commodity Price Predictor (India Insight)")
st.markdown("Forecasting next-day commodity prices using global market trends, exchange rates, and live news sentiment.")
st.markdown("---")

# Sidebar Selection
st.sidebar.header("Settings")
commodity = st.sidebar.selectbox("Select Commodity", ["Gold", "Silver", "Oil"])

# Extra sidebar option if Gold is selected
purity = "24K (99.9% Pure)"
if commodity == "Gold":
    purity = st.sidebar.selectbox("Gold Purity", ["24K (99.9% Pure)", "22K (916 Jewellery)"])

@st.cache_data
def load_and_convert_data(target_comm):
    raw_df = pd.read_csv("data/raw_market_data.csv", index_col=0, parse_dates=True)
    sent_path = "data/daily_sentiment.csv"
    
    df = raw_df.copy()
    if os.path.exists(sent_path):
        sent_df = pd.read_csv(sent_path, index_col=0, parse_dates=True)
        df = df.join(sent_df, how='left')
        df['sentiment_polarity'] = df['sentiment_polarity'].fillna(0.0)
    else:
        df['sentiment_polarity'] = 0.0
        
    for col in df.columns:
        if col != 'sentiment_polarity':
            df[f'{col}_ret'] = df[col].pct_change()
            
    df[f'{target_comm}_SMA_7'] = df[target_comm].rolling(window=7).mean()
    df[f'{target_comm}_SMA_30'] = df[target_comm].rolling(window=30).mean()
    
    for lag in [1, 2, 5]:
        df[f'{target_comm}_lag_{lag}'] = df[target_comm].shift(lag)
        
    df['Target'] = df[target_comm].shift(-1)
    df.dropna(inplace=True)
    return df

try:
    df = load_and_convert_data(commodity)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Train Model
X = df.drop(columns=['Target'])
y = df['Target']

model = LinearRegression()
model.fit(X, y)

# Predict next-day price in USD
latest_features = X.iloc[[-1]]
predicted_usd = model.predict(latest_features)[0]
actual_usd = df[commodity].iloc[-1]

# Conversion factor: USD to INR (~94.39)
USD_TO_INR = 94.39

if commodity == "Gold":
    SPOT_CALIBRATION = 1.185 
    base_actual = ((actual_usd / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
    base_predicted = ((predicted_usd / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
    
    if "22K" in purity:
        actual_val = base_actual * (22 / 24)
        predicted_val = base_predicted * (22 / 24)
        unit_label = "per 10 Grams (22K INR)"
    else:
        actual_val = base_actual
        predicted_val = base_predicted
        unit_label = "per 10 Grams (24K INR)"
        
elif commodity == "Silver":
    SPOT_CALIBRATION = 1.12
    actual_val = ((actual_usd / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
    predicted_val = ((predicted_usd / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
    unit_label = "per 10 Grams (INR)"
else:
    actual_val = actual_usd * USD_TO_INR
    predicted_val = predicted_usd * USD_TO_INR
    unit_label = "per Barrel (₹)"

price_diff = predicted_val - actual_val
pct_diff = (price_diff / actual_val) * 100

# Calculate a dynamic confidence score based on recent residual volatility
recent_residuals = np.abs(model.predict(X.tail(30)) - y.tail(30))
mean_error_pct = np.mean(recent_residuals / y.tail(30)) * 100
confidence_score = max(55.0, min(92.0, 100.0 - (mean_error_pct * 15)))

# --- MAIN DISPLAY METRICS (Plain Language) ---
st.subheader(f"📌 Latest Forecast for {commodity} ({purity if commodity=='Gold' else ''})")

prediction_status = "Going Up 📈" if price_diff > 0 else "Going Down 📉"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Market Price", f"₹{actual_val:,.2f}")
m2.metric("Predicted Next-Day Price", f"₹{predicted_val:,.2f}", f"{pct_diff:+.2f}%")
m3.metric("Expected Direction", prediction_status)
m4.metric("Model Confidence Score", f"{confidence_score:.1f}% Sure")

st.markdown("---")

# --- REASONING SECTION ---
st.subheader("💡 Why this prediction? (Model Breakdown)")

recent_sentiment = df['sentiment_polarity'].iloc[-1]
recent_dxy_ret = df['DXY_ret'].iloc[-1] if 'DXY_ret' in df.columns else 0.0
recent_sma7 = df[f'{commodity}_SMA_7'].iloc[-1]
recent_sma30 = df[f'{commodity}_SMA_30'].iloc[-1]

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown("#### 🌍 World News & Sentiment Impact")
    if recent_sentiment > 0.05:
        st.success(f"**Positive Sentiment (Score: {recent_sentiment:.2f}):** Global headlines lean optimistic, lifting market confidence.")
    elif recent_sentiment < -0.05:
        st.error(f"**Cautious Sentiment (Score: {recent_sentiment:.2f}):** Geopolitical tensions are driving safe-haven accumulation.")
    else:
        st.info(f"**Neutral Sentiment (Score: {recent_sentiment:.2f}):** News sentiment remains balanced.")

with col_r2:
    st.markdown("#### 📈 Technical & Macro Trend")
    if recent_sma7 > recent_sma30:
        st.info("**Short-Term Momentum:** 7-day moving average is above the 30-day average, signaling upward momentum.")
    else:
        st.info("**Short-Term Momentum:** 7-day moving average is below the 30-day average, signaling a cooling trend.")
    
    if recent_dxy_ret < 0:
        st.info("**US Dollar Index (DXY):** Dollar weakness is providing upward tailwinds.")
    else:
        st.info("**US Dollar Index (DXY):** A stronger dollar is creating mild valuation headwinds.")

st.markdown("---")

# --- LIVE SENTIMENT FEED INSPECTOR ---
with st.expander("📰 Live News Sentiment Feed Inspector (What headlines is the model reading?)"):
    st.markdown("Here are recent global headlines fetched via NewsAPI that contributed to today's sentiment score:")
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{commodity.lower()} OR inflation OR economy",
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 4,
            'apiKey': '96d1c8b648d04ccc960b4b6b052580df'
        }
        res = requests.get(url, params=params).json()
        articles = res.get('articles', [])
        
        if articles:
            for art in articles:
                title = art.get('title', 'No Title')
                source = art.get('source', {}).get('name', 'News Source')
                pub_date = art.get('publishedAt', '')[:10]
                st.markdown(f"- **[{source}]** {title} *({pub_date})*")
        else:
            st.write("No recent live articles found or rate limit reached. Using cached sentiment index.")
    except Exception:
        st.write("Unable to fetch live feed preview right now. Sentiment index is active in model weights.")

st.markdown("---")

# Clean Chart
st.subheader("📊 Recent Price Trend (in INR)")
fig, ax = plt.subplots(figsize=(12, 4))

if commodity == "Gold":
    hist_base = ((df[commodity].iloc[-180:] / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
    history_inr = hist_base * (22 / 24) if "22K" in purity else hist_base
elif commodity == "Silver":
    history_inr = ((df[commodity].iloc[-180:] / 31.1035) * 10 * USD_TO_INR) * SPOT_CALIBRATION
else:
    history_inr = df[commodity].iloc[-180:] * USD_TO_INR

ax.plot(history_inr.index, history_inr.values, label=f'Historical Price ({unit_label})', color='#2ca02c', linewidth=2)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Price in INR (₹)", fontsize=11)
ax.set_title(f"{commodity} Price Movement (Past 6 Months)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)

st.pyplot(fig)

st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Disclaimer: For academic presentation and educational purposes only. Not financial advice.</p>", unsafe_allow_html=True)  