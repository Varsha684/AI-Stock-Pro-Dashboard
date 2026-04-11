import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf  

from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder

from src.model import MyLSTMModel

# --- 1. PAGE SETUP & GLASSMORPHISM CSS ---
st.set_page_config(page_title="AI Stock Pro", layout="wide", page_icon="⚡")

page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f19;
    background-image: radial-gradient(circle at 15% 50%, rgba(20, 255, 236, 0.08), transparent 25%),
                      radial-gradient(circle at 85% 30%, rgba(255, 42, 112, 0.08), transparent 25%);
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stMetricValue"] {
    color: #14FFEC !important;
    text-shadow: 0px 0px 10px rgba(20, 255, 236, 0.4);
    font-size: 2.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 1.1rem !important;
}
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# --- 2. PREMIUM NAVIGATION BAR ---
selected_tab = option_menu(
    menu_title=None,
    options=["Live Market AI", "Raw Data Analytics", "Model Settings"],
    icons=["bar-chart-line-fill", "table", "gear-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1A202C", "border-radius": "10px"},
        "icon": {"color": "#14FFEC", "font-size": "18px"},
        "nav-link": {"color": "#E2E8F0", "font-size": "16px", "text-align": "center", "margin": "0px"},
        "nav-link-selected": {"background-color": "#2D3748", "color": "#14FFEC", "font-weight": "bold", "border-bottom": "2px solid #14FFEC"},
    }
)

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.markdown("### ⚡ **AI Engine Controls**")
stock_options = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS (Tata Consultancy)": "TCS.NS",
    "Infosys": "INFY.NS",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Nvidia": "NVDA"
}
selected_stock_name = st.sidebar.selectbox("🎯 Select Asset", list(stock_options.keys()))
stock_symbol = stock_options[selected_stock_name]

days_to_predict = st.sidebar.slider("🔮 Prediction Horizon (Days)", 1, 60, 14)
seq_length = 10 

# --- 4. CORE LOGIC ---
@st.cache_data 
def load_live_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

data = load_live_data(stock_symbol)

model = MyLSTMModel(1, 32, 1, 1)
try:
    model.load_state_dict(torch.load('lstm_stock_model.pth', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    st.error("Model file missing!")
    st.stop()

def get_future_predictions(model, scaler, raw_data, seq_length, days):
    prices = raw_data['Close'].values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    current_seq = torch.FloatTensor(scaled_prices[-seq_length:]).unsqueeze(0) 
    future_preds = []
    with torch.no_grad():
        for _ in range(days):
            pred = model(current_seq) 
            future_preds.append(pred.item())
            current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data['Close'].values.reshape(-1, 1))
future_prices = get_future_predictions(model, scaler, data, seq_length, days_to_predict)

currency = "₹" if ".NS" in stock_symbol else "$"
current_price = float(np.asarray(data['Close'])[-1])
predicted_tomorrow = float(future_prices[0][0])
price_change = predicted_tomorrow - current_price
change_percent = (price_change / current_price) * 100

# --- 5. TAB VIEWS ---

if selected_tab == "Live Market AI":
    # 1. Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("LTP (Last Traded Price)", f"{currency}{current_price:.2f}")
    col2.metric("AI Target (24h)", f"{currency}{predicted_tomorrow:.2f}", f"{price_change:.2f} ({change_percent:.2f}%)")
    col3.metric("AI Signal", "STRONG BUY 🚀" if price_change > 0 else "SELL ⚠️", delta="Bullish" if price_change > 0 else "Bearish")
    st.markdown("<hr style='border: 1px solid #2D3748;'>", unsafe_allow_html=True)

    # 2. Graph
    past_days = 90
    recent_data = data.tail(past_days).copy()
    future_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=days_to_predict)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=recent_data['Date'], open=recent_data['Open'], high=recent_data['High'], low=recent_data['Low'], close=recent_data['Close'], name='Market Data'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices.flatten(), mode='lines+markers', name='AI Neural Forecast', line=dict(color='#14FFEC', width=3, dash='dot')))

    fig.update_layout(
        title=dict(text=f"<b>{selected_stock_name}</b> - Institutional AI Forecast", font=dict(size=20, color="#E2E8F0")),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#2D3748'),
        template="plotly_dark", height=500, xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11, 15, 25, 0.8)',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Screen ko do hisso me baantna tables ke liye
    col_left, col_right = st.columns(2)
    
    with col_left:
        # 3. FUTURE PREDICTIONS TABLE 
        st.markdown(f"### 🔮 AI Predicted Prices")
        prediction_df = pd.DataFrame({
            "Date": future_dates.strftime('%Y-%m-%d'),
            "Predicted Target Price": [f"{currency}{val:.2f}" for val in future_prices.flatten()]
        })
        st.dataframe(prediction_df, use_container_width=True, height=300)

    with col_right:
        # 4. RECENT RAW DATA TABLE (Wapas add kiya gaya)
        st.markdown(f"### 📊 Recent Raw Data")
        recent_raw = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).sort_values(by='Date', ascending=False)
        recent_raw['Date'] = recent_raw['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(recent_raw, use_container_width=True, height=300)

elif selected_tab == "Raw Data Analytics":
    st.markdown(f"### 📊 Deep Analytics for {selected_stock_name}")
    st.caption("Interact with the table below: Filter, sort, or export data like a Pro.")
    
    display_df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(100).sort_values(by='Date', ascending=False)
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    for col in ['Open', 'High', 'Low', 'Close']:
        display_df[col] = display_df[col].round(2)

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar() 
    gridOptions = gb.build()

    AgGrid(display_df, gridOptions=gridOptions, theme='alpine', height=500, fit_columns_on_grid_load=True)

elif selected_tab == "Model Settings":
    st.markdown("### 🧠 AI Engine Architecture")
    st.info("Current Model: Long Short-Term Memory (LSTM) Neural Network")