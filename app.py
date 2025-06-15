
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Coca-Cola Stock Dashboard", layout="wide")

st.title("📈 Coca-Cola Stock Price Prediction App")

@st.cache_data
def load_data():
    ticker = 'KO'
    data = yf.download(ticker, start='2015-01-01', end='2023-12-31')
    data.reset_index(inplace=True)
    return data

data = load_data()
st.subheader("Raw Coca-Cola Stock Data")
st.write(data.tail())

# Feature Engineering
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

# Line Chart
st.subheader("📊 Closing Price & Moving Averages")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data['Date'], data['Close'], label='Close')
ax.plot(data['Date'], data['MA_20'], label='MA 20')
ax.plot(data['Date'], data['MA_50'], label='MA 50')
ax.set_title('Coca-Cola Stock Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# ML Model Training
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
X = data[features]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Live Prediction
st.subheader("🔮 Predicting Today's Close Price (Live)")
live_data = yf.download('KO', period='1d', interval='1m')
live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
live_data.fillna(0, inplace=True)

try:
    latest = live_data[features].dropna().iloc[-1:]
    prediction = model.predict(latest)[0]
    st.success(f"✅ Predicted Closing Price: **${prediction:.2f}**")
except Exception as e:
    st.error("Live data is incomplete, try again in a few minutes.")

st.markdown("---")
st.caption("Made with ❤️ by Anjali – Coca-Cola Stock Prediction Dashboard")
