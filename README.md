# Stock Price Predictor (Student Project)

This is my student project for predicting stock prices using Machine Learning (LSTM neural networks). It also has a Streamlit web app dashboard!

## Features included:
1. **Fetching data**: Uses `yfinance` to grab stock data from Yahoo Finance.
2. **Indicators**: Calculates 50-day and 200-day Moving Averages, and RSI (Relative Strength Index).
3. **Deep Learning**: Uses an LSTM neural network built with Keras/TensorFlow.
4. **Dashboard**: A Streamlit app to view the data and predictions.

## How to run this project:

### 1. Install dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model & Generate Graphs
First, you need to run the Jupyter Notebook to train the model and save its weights.
```bash
jupyter notebook Stock_Prediction_LSTM.ipynb
```
- Open the notebook.
- Run all the cells.
- It will train the model, plot the graphs, and save `student_lstm_model.h5`.

### 3. Run the Streamlit Dashboard
Once the `student_lstm_model.h5` file is created, run the Streamlit app:
```bash
streamlit run app.py
```
- Open the link provided in your browser.
- Enter a stock ticker and see the predictions!
