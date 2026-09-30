import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import os

# --- Title and Description ---
st.title("📈 Stock Price Predictor - Student Project")
st.write("Welcome to my project! I built an LSTM Neural Network to predict stock trends based on past data.")

# --- Sidebar Inputs ---
st.sidebar.header("User Input")
stock_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOG, TSLA):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

if st.sidebar.button("Fetch & Predict"):
    
    st.subheader(f"Data for {stock_symbol}")
    st.write("Fetching data using yfinance...")
    
    # 1. Fetch data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if len(data) == 0:
        st.error("No data found! Please check the ticker symbol and dates.")
    else:
        st.write(data.tail())
        
        # 2. Moving Averages & RSI
        st.subheader("Technical Indicators")
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        
        # Plotting MA
        fig_ma, ax_ma = plt.subplots(figsize=(10, 5))
        ax_ma.plot(data.index, data['Close'], label='Close Price')
        ax_ma.plot(data.index, data['MA_50'], label='50-day MA', color='red')
        ax_ma.plot(data.index, data['MA_200'], label='200-day MA', color='green')
        ax_ma.set_title("Price with Moving Averages")
        ax_ma.legend()
        st.pyplot(fig_ma)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Plotting RSI
        fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
        ax_rsi.plot(data.index, data['RSI'], label='RSI (14 days)', color='purple')
        ax_rsi.axhline(70, linestyle='--', color='red', alpha=0.5)  # Overbought
        ax_rsi.axhline(30, linestyle='--', color='green', alpha=0.5) # Oversold
        ax_rsi.set_title("Relative Strength Index (RSI)")
        ax_rsi.legend()
        st.pyplot(fig_rsi)

        # 3. Model Prediction
        st.subheader("LSTM Model Prediction vs Actual")
        
        model_path = 'student_lstm_model.h5'
        
        if not os.path.exists(model_path):
            st.warning("⚠️ The model weights file `student_lstm_model.h5` is missing!")
            st.info("💡 Hint: You need to run all the cells in the Jupyter Notebook (`Stock_Prediction_LSTM.ipynb`) first so it can train and save the model file.")
        else:
            try:
                st.write("Loading pre-trained student model...")
                model = load_model(model_path)
                
                # Drop NaN to align with what the model expects
                clean_data = data.dropna()
                dataset = clean_data.filter(['Close']).values
                
                # Need at least 60 days to predict
                if len(dataset) <= 60:
                    st.error("Not enough data to make predictions! Need more than 60 days.")
                else:
                    training_data_len = math.ceil(len(dataset) * 0.8)
                    
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(dataset)
                    
                    test_data = scaled_data[training_data_len - 60: , :]
                    x_test = []
                    for i in range(60, len(test_data)):
                        x_test.append(test_data[i-60:i, 0])
                    
                    x_test = np.array(x_test)
                    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                    
                    predictions = model.predict(x_test)
                    predictions = scaler.inverse_transform(predictions)
                    
                    valid = clean_data.filter(['Close'])[training_data_len:]
                    valid['Predictions'] = predictions
                    
                    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                    ax_pred.plot(valid.index, valid['Close'], label='Actual Price')
                    ax_pred.plot(valid.index, valid['Predictions'], label='Predicted Price', color='orange')
                    ax_pred.set_title("Test Set Predictions")
                    ax_pred.legend()
                    st.pyplot(fig_pred)
                    
            except Exception as e:
                st.error(f"Something went wrong while running the model: {e}")
