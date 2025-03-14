# S&P 500 Financial Forecasting with LSTM

I built a Long Short-Term Memory (LSTM) model to forecast S&P 500 stock prices using historical data. The process involved data preprocessing, feature engineering, sequence preparation, and training the LSTM to predict future closing prices.

### 1. Data Loading and Preprocessing
I started by importing the S&P 500 dataset and filtering it down to the most recent 15,000 records. I cleaned any NaN or infinite values and converted the `date` column into datetime format. To standardize the price data, I scaled the closing prices between 0 and 1.

### 2. Feature Engineering
I created new characteristics within the dataset:
- **Price Change**: Computed the percentage difference between opening and closing prices.
- **Technical Indicators**: Added RSI, MACD, and Bollinger Bands.
- **Moving Averages**: Included both short-term (5-day) and medium-term (20-day) moving averages.
- **Temporal and Interaction Features**: Incorporated the day of the week, month, quarter, and interactions between price and volume.

These enhancements helped the model capture a broader range of market behavior.

### 3. Sequence Preparation
I then formed sequences from the prior 60 days to predict the subsequent day’s price. After creating these sequences, I converted them into PyTorch tensors, making them ready for ingestion by the LSTM model.

### 4. Model Training and Results
I configured an LSTM network with a single hidden layer and a fully connected output layer. I used MSE (Mean Squared Error) as the loss function and Adam as the optimizer. Training took place on two NVIDIA T4 Tensor Core GPUs over 100 epochs, showing improvements:

- **Initial Loss**: 10.3%
- **Loss after 10 Epochs**: 4.1%
- **Loss after 30 Epochs**: 1.1%
- **Final Loss after 100 Epochs**: 0.06%
- **Approximate Accuracy**: 93%

This accuracy shows that the model learned historical patterns within the data.

### 5. Prediction and Visualization
After model training, I set the model to evaluation mode to generate predictions. After reversing the scaling, I plotted these predictions against the actual closing prices, showing the model’s predictive strength.

### Dependencies
- **Python Libraries**: `pandas`, `numpy`, `torch`, `sklearn`, `matplotlib`
