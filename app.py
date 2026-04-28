import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")

st.title("📈 FinSight")
st.markdown("Enter any stock ticker to get an AI-powered prediction + news sentiment analysis.")

# ── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("🔧 Settings")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
market = st.sidebar.selectbox("Market", ["US (NASDAQ/NYSE)", "India (NSE)"])
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
news_api_key = st.sidebar.text_input("NewsAPI Key (optional - free at newsapi.org)", type="password")

if market == "India (NSE)":
    ticker = ticker + ".NS"

st.sidebar.markdown("---")
st.sidebar.markdown("**Example US tickers:** AAPL, TSLA, MSFT, GOOGL")
st.sidebar.markdown("**Example India tickers:** RELIANCE, TCS, INFY, HDFCBANK")

# ── Load Stock Data ───────────────────────────────────────────
@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

if st.button("🚀 Run Prediction", type="primary"):
    with st.spinner("Fetching stock data..."):
        df = load_data(ticker, period)

    if df.empty:
        st.error("❌ Could not find data for this ticker. Please check the symbol and try again.")
        st.stop()

    # ── Feature Engineering ───────────────────────────────────
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # ── Model Training ────────────────────────────────────────
    features = ['SMA_20', 'SMA_50', 'Return', 'Volatility', 'Momentum']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    latest_features = X.iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    confidence = model.predict_proba(latest_features)[0][prediction]

    # ── News Sentiment ────────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = 0
    news_items = []

    clean_ticker = ticker.replace(".NS", "")

    if news_api_key:
        try:
            url = f"https://newsapi.org/v2/everything?q={clean_ticker}+stock&language=en&sortBy=publishedAt&pageSize=5&apiKey={news_api_key}"
            response = requests.get(url)
            articles = response.json().get("articles", [])
            for a in articles:
                score = analyzer.polarity_scores(a['title'])['compound']
                news_items.append({"headline": a['title'], "score": score, "url": a['url']})
            if news_items:
                sentiment_score = np.mean([n['score'] for n in news_items])
        except:
            st.warning("Could not fetch news. Continuing without sentiment.")
    else:
        st.info("💡 Add a free NewsAPI key in the sidebar to enable live news sentiment.")

    # ── Layout: Metrics Row ───────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    price_change = current_price - prev_price
    price_pct = (price_change / prev_price) * 100

    col1.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_pct:+.2f}%)")
    col2.metric("🎯 Model Accuracy", f"{accuracy*100:.1f}%")
    col3.metric("🔮 Tomorrow's Prediction", "📈 UP" if prediction == 1 else "📉 DOWN", f"Confidence: {confidence*100:.1f}%")

    if news_items:
        sentiment_label = "😊 Positive" if sentiment_score > 0.05 else ("😟 Negative" if sentiment_score < -0.05 else "😐 Neutral")
        col4.metric("📰 News Sentiment", sentiment_label, f"Score: {sentiment_score:.2f}")
    else:
        col4.metric("📰 News Sentiment", "N/A", "Add API key")

    # ── Price Chart ───────────────────────────────────────────
    st.markdown("### 📊 Price Chart with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00b4d8', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#f77f00', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#e63946', width=1.5, dash='dash')))
    fig.update_layout(
        template='plotly_dark',
        height=450,
        xaxis_title="Date",
        yaxis_title="Price (USD)" if ".NS" not in ticker else "Price (INR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────
    st.markdown("### 🧠 What Drives the Prediction?")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig2 = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#06d6a0'
    ))
    fig2.update_layout(template='plotly_dark', height=300, xaxis_title="Importance Score")
    st.plotly_chart(fig2, use_container_width=True)

    # ── News Section ──────────────────────────────────────────
    if news_items:
        st.markdown("### 📰 Latest News Sentiment")
        for item in news_items:
            color = "🟢" if item['score'] > 0.05 else ("🔴" if item['score'] < -0.05 else "🟡")
            st.markdown(f"{color} [{item['headline']}]({item['url']}) — Score: `{item['score']:.2f}`")

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only. Not financial advice. Always do your own research before investing.")
