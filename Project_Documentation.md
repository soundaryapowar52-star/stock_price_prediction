# Comprehensive Project Documentation: Stock Price Trend Prediction with LSTM

## 1. Project Overview
**Project Title:** Stock Price Trend Prediction using Long Short-Term Memory (LSTM) Networks  
**Type:** Academic / Student Project

### 1.1 Abstract
Predicting stock prices is a challenging task due to the highly volatile and non-linear nature of financial markets. This project aims to predict future stock prices using historical price trends. It leverages Deep Learning, specifically Long Short-Term Memory (LSTM) neural networks, which are highly effective for time-series forecasting. The project also integrates financial technical indicators such as Moving Averages (MA) and the Relative Strength Index (RSI) to provide a comprehensive analysis.

### 1.2 Objectives
- To collect and preprocess historical stock market data using the Yahoo Finance API.
- To compute and visualize key technical indicators (50-day MA, 200-day MA, RSI).
- To design, train, and validate an LSTM neural network for predicting closing stock prices.
- To deploy a user-friendly web dashboard using Streamlit to interact with the trained model and view data insights.

---

## 2. Tools & Technologies Used
- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Deep Learning:** TensorFlow, Keras, Scikit-learn
- **Data Visualization:** Matplotlib
- **Data Source:** `yfinance` (Yahoo Finance API)
- **Deployment/Web App:** Streamlit
- **Development Environment:** Jupyter Notebook

---

## 3. Methodology

### 3.1 Data Collection
Historical stock data is retrieved programmatically using the `yfinance` library. The dataset includes Open, High, Low, Close prices, and trading Volume. The primary feature used for training the predictive model is the **Close** price.

### 3.2 Feature Engineering & Technical Indicators
To better understand market trends, the following indicators were engineered:
- **50-Day Moving Average (MA):** Represents short-term price trends and smooths out daily price fluctuations.
- **200-Day Moving Average (MA):** Represents long-term price trends. The intersection of 50-day and 200-day MAs is often used by traders to identify major market shifts.
- **Relative Strength Index (RSI):** A momentum oscillator measuring the speed and change of price movements (calculated on a 14-day window). Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.

### 3.3 Data Preprocessing
- **Handling Missing Values:** NaN values introduced by rolling windows (like moving averages) are removed to maintain data integrity.
- **Normalization:** Neural networks converge faster and perform better when data is scaled. The closing prices are scaled to a range between 0 and 1 using Scikit-learn's `MinMaxScaler`.
- **Sequence Generation:** Since LSTMs require sequential data, the dataset is transformed into time steps. The model looks back at the past 60 days of closing prices to predict the price on the 61st day.
- **Train/Test Split:** The dataset is split chronologically. The first 80% of the historical data is used for training, and the remaining 20% is held back for testing to evaluate how well the model generalizes to unseen future data.

### 3.4 Model Architecture
The model uses a Sequential architecture built with Keras:
1. **LSTM Layer 1:** 50 units, `return_sequences=True` (passes the sequence output to the next LSTM layer).
2. **LSTM Layer 2:** 50 units, `return_sequences=False` (only returns the final output of the sequence).
3. **Dense Layer 1:** 25 units (fully connected layer to capture complex relationships).
4. **Dense Layer 2:** 1 unit (final output layer producing the predicted price).

- **Optimizer:** Adam (Adaptive Moment Estimation)
- **Loss Function:** Mean Squared Error (MSE)

### 3.5 Model Training
The model is trained on the prepared sequences over a set number of epochs (defaulted to 5 for quicker execution) with a batch size of 32. After training, the model weights are saved to an `.h5` file (`student_lstm_model.h5`) so they can be loaded instantly by the web application without retraining.

---

## 4. Web Dashboard (Streamlit)
To make the project interactive and accessible, a Streamlit application (`app.py`) was developed. The app allows users to:
- Input any valid stock ticker symbol (e.g., AAPL, TSLA, GOOG).
- Select custom start and end dates.
- Visualize historical closing prices overlaid with 50-day and 200-day moving averages.
- View the RSI momentum chart.
- Compare the LSTM model's predictions against the actual historical test data via interactive plots.

---

## 5. Deliverables & Project Structure
The project folder consists of the following key files:
1. **`Stock_Prediction_LSTM.ipynb`**: The primary Jupyter Notebook containing the step-by-step code, markdown explanations, model training, and evaluation graphs.
2. **`app.py`**: The Streamlit web application script.
3. **`student_lstm_model.h5`**: The saved weights of the trained LSTM model (generated after running the notebook).
4. **`requirements.txt`**: A list of all Python dependencies required to run the project.
5. **`README.md`**: A quick-start guide for setting up the project.

---

## 6. How to Run the Project
1. **Install Dependencies:**
   Open a terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model:**
   Launch Jupyter Notebook and run all cells in `Stock_Prediction_LSTM.ipynb`. This will download the dataset, train the model, plot the evaluation metrics, and generate the `student_lstm_model.h5` file.
3. **Launch the Dashboard:**
   In the terminal, execute:
   ```bash
   streamlit run app.py
   ```
   Access the interactive dashboard via the local web address provided in your terminal.

---

## 7. Conclusion
This project successfully demonstrates the application of Deep Learning for financial time-series forecasting. While predicting the stock market with absolute certainty is impossible, LSTM networks prove to be highly capable of identifying underlying trends and patterns in historical price data. The addition of technical indicators (MA and RSI) provides supplementary insights that are highly valuable in real-world trading analysis. This project serves as an excellent foundational study into AI-driven quantitative finance.
