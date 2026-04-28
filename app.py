import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #00b4d8, #06d6a0, #f77f00);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card { background: #1e1e2e; border-radius: 12px; padding: 1rem; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">📈 FinSight AI — Stock Predictor</p>', unsafe_allow_html=True)
st.markdown("Enter any stock ticker to get an AI-powered prediction + news sentiment analysis.")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🔧 Settings")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
market = st.sidebar.selectbox("Market", ["US (NASDAQ/NYSE)", "India (NSE)"])
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
news_api_key = st.sidebar.text_input("NewsAPI Key (optional - free at newsapi.org)", type="password")

is_indian = market == "India (NSE)"
currency_symbol = "₹" if is_indian else "$"

if is_indian:
    ticker_full = ticker + ".NS"
else:
    ticker_full = ticker

st.sidebar.markdown("---")
st.sidebar.markdown("**🇺🇸 US tickers:** AAPL, TSLA, MSFT, GOOGL")
st.sidebar.markdown("**🇮🇳 India tickers:** RELIANCE, TCS, INFY, HDFCBANK")

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data(ticker_full, period):
    df = yf.download(ticker_full, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if st.button("🚀 Run Prediction", type="primary"):

    with st.spinner("⏳ Fetching stock data..."):
        df = load_data(ticker_full, period)

    if df.empty:
        st.error("❌ Could not find data. Please check the ticker symbol and try again.")
        st.stop()

    # ── Features ──────────────────────────────────────────────
    df['SMA_20']     = df['Close'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    df['EMA_20']     = df['Close'].ewm(span=20).mean()
    df['Return']     = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Momentum']   = df['Close'] - df['Close'].shift(5)
    df['RSI']        = 100 - (100 / (1 + df['Return'].clip(lower=0).rolling(14).mean() /
                               (-df['Return'].clip(upper=0).rolling(14).mean()).replace(0, 1e-10)))
    df['Volume_MA']  = df['Volume'].rolling(20).mean()
    df['Target']     = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # ── Model ─────────────────────────────────────────────────
    features = ['SMA_20','SMA_50','EMA_20','Return','Volatility','Momentum','RSI']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(n_estimators=200, max_depth=4, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds       = model.predict(X_test)
    accuracy    = accuracy_score(y_test, preds)
    prediction  = model.predict(X.iloc[[-1]])[0]
    confidence  = model.predict_proba(X.iloc[[-1]])[0][prediction]

    # ── Sentiment ─────────────────────────────────────────────
    analyzer       = SentimentIntensityAnalyzer()
    sentiment_score = 0
    news_items     = []
    clean_ticker   = ticker

    if news_api_key:
        try:
            url = f"https://newsapi.org/v2/everything?q={clean_ticker}+stock&language=en&sortBy=publishedAt&pageSize=6&apiKey={news_api_key}"
            articles = requests.get(url).json().get("articles", [])
            for a in articles:
                score = analyzer.polarity_scores(a['title'])['compound']
                news_items.append({"headline": a['title'], "score": score, "url": a['url']})
            if news_items:
                sentiment_score = np.mean([n['score'] for n in news_items])
        except:
            st.warning("Could not fetch news.")
    else:
        st.info("💡 Add a free NewsAPI key in the sidebar to enable live news sentiment.")

    # ── KPI Row ───────────────────────────────────────────────
    st.markdown("---")
    current_price = float(df['Close'].iloc[-1])
    prev_price    = float(df['Close'].iloc[-2])
    price_change  = current_price - prev_price
    price_pct     = (price_change / prev_price) * 100
    high_52w      = float(df['High'].max())
    low_52w       = float(df['Low'].min())
    avg_volume    = int(df['Volume'].mean())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Current Price",      f"{currency_symbol}{current_price:.2f}", f"{price_change:+.2f} ({price_pct:+.2f}%)")
    col2.metric("🎯 Model Accuracy",     f"{accuracy*100:.1f}%")
    col3.metric("🔮 Tomorrow",           "📈 UP" if prediction == 1 else "📉 DOWN", f"Confidence: {confidence*100:.1f}%")
    col4.metric("📅 52W High / Low",     f"{currency_symbol}{high_52w:.2f}", f"Low: {currency_symbol}{low_52w:.2f}")
    col5.metric("📊 Avg Volume",         f"{avg_volume:,}")

    # ── Chart 1: Candlestick + MAs ────────────────────────────
    st.markdown("### 🕯️ Candlestick Chart with Moving Averages")
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig1.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price',
        increasing_line_color='#06d6a0', decreasing_line_color='#e63946'), row=1, col=1)

    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],  name='SMA 20',
        line=dict(color='#f77f00', width=1.5, dash='dot')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],  name='SMA 50',
        line=dict(color='#00b4d8', width=1.5, dash='dash')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df.index, y=df['EMA_20'],  name='EMA 20',
        line=dict(color='#ff006e', width=1.5)), row=1, col=1)

    colors = ['#06d6a0' if v >= df['Volume_MA'].iloc[i] else '#e63946'
              for i, v in enumerate(df['Volume'])]
    fig1.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
        marker_color=colors, opacity=0.7), row=2, col=1)

    fig1.update_layout(template='plotly_dark', height=550,
        yaxis_title=f"Price ({currency_symbol})", yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: RSI ──────────────────────────────────────────
    st.markdown("### 📉 RSI — Relative Strength Index")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#f77f00', width=2), fill='tozeroy',
        fillcolor='rgba(247,127,0,0.15)'))
    fig2.add_hline(y=70, line_dash="dash", line_color="#e63946",
        annotation_text="Overbought (70)")
    fig2.add_hline(y=30, line_dash="dash", line_color="#06d6a0",
        annotation_text="Oversold (30)")
    fig2.update_layout(template='plotly_dark', height=300,
        yaxis=dict(range=[0, 100]), xaxis_title="Date", yaxis_title="RSI")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Returns Distribution ────────────────────────
    st.markdown("### 📊 Daily Returns Distribution")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df['Return'], nbinsx=60,
        marker=dict(color='#00b4d8', line=dict(color='#ffffff', width=0.3)),
        name='Daily Return', opacity=0.85))
    fig3.add_vline(x=0, line_color='white', line_dash='dash')
    fig3.update_layout(template='plotly_dark', height=320,
        xaxis_title="Daily Return", yaxis_title="Frequency")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Volatility ───────────────────────────────────
    st.markdown("### 🌊 Rolling Volatility (10-Day)")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df['Volatility'],
        name='Volatility', line=dict(color='#ff006e', width=2),
        fill='tozeroy', fillcolor='rgba(255,0,110,0.15)'))
    fig4.update_layout(template='plotly_dark', height=300,
        xaxis_title="Date", yaxis_title="Volatility")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 5: Feature Importance ───────────────────────────
    st.markdown("### 🧠 What Drives the AI Prediction?")
    imp_df = pd.DataFrame({'Feature': features,
                           'Importance': model.feature_importances_}
                         ).sort_values('Importance', ascending=True)
    fig5 = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'],
        orientation='h',
        marker=dict(color=imp_df['Importance'],
                    colorscale='Viridis', showscale=True)))
    fig5.update_layout(template='plotly_dark', height=350,
        xaxis_title="Importance Score")
    st.plotly_chart(fig5, use_container_width=True)

    # ── News ──────────────────────────────────────────────────
    if news_items:
        st.markdown("### 📰 Latest News Sentiment")
        for item in news_items:
            color = "🟢" if item['score'] > 0.05 else ("🔴" if item['score'] < -0.05 else "🟡")
            st.markdown(f"{color} [{item['headline']}]({item['url']}) — Score: `{item['score']:.2f}`")

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice. Always do your own research.")
