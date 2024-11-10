# S&P 500 Financial Forecasting with LSTM README

This project utilizes a Long Short-Term Memory (LSTM) neural network to forecast S&P 500 stock prices based on historical data. The workflow includes data preprocessing, feature engineering, sequence creation for model input, and training an LSTM to predict future prices.

## Project Steps

### 1. Data Loading and Preprocessing
- Load the S&P 500 dataset and focus on the last 15,000 entries.
- Remove NaN and infinite values, convert the date column to datetime, and scale the closing prices to a range of 0-1 using `MinMaxScaler`.

### 2. Feature Engineering
- Generate new features to improve model performance:
  - **Price Change, High-Low Range, Volume Change**: Derived from raw data.
  - **Moving Averages (MA5, MA10, MA20)**: Capture price trends.
  - **Technical Indicators**: Includes RSI (14), EMA (12, 26), MACD, Bollinger Bands, VWAP.
  - **Lagged Features**: Previous closing prices (1, 2, 5-day lags).
  - **Temporal Features**: Day of the week, month, and quarter.
  - **Interaction Features**: Price-volume interaction and high-low ratio.

### 3. Sequence Preparation
- Prepare sequences of the last 60 days' data as model input to predict the next day's price.
- Convert the sequences and labels to PyTorch tensors.

### 4. Model Definition and Training
- Define an LSTM model with one hidden layer and a fully connected output layer.
- Train the model using Mean Squared Error (MSE) loss and Adam optimizer on GPU if available.
- The model trains for 100 epochs with loss outputs every 10 epochs.

### 5. Prediction and Visualization
- Switch the model to evaluation mode to generate predictions.
- Reverse the scaling of predictions for interpretability.
- Plot true vs. predicted prices to visually assess model performance.

### Dependencies
- Python libraries: `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`

### Usage
1. Execute the code sequentially in a compatible environment (e.g., Kaggle, Colab).
2. After training, visualize the predictions compared to actual closing prices for the forecasted period.

This project demonstrates a simple LSTM-based approach to time-series forecasting for financial data.
