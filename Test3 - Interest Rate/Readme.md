# Short description: ​

Using a LSTM for predicting the DAX Index whether it is profitable to buy on a given day or not. To improve accuracy the current ecb interest rate was added as data.

## Modeling architecture

- LSTM

### Model parameter

- input_size = 9
- output_size = 1
- hidden_size = 1000
- num_layers = 6
- dropout = 0.2

### Training parameter

- batch_size = 16
- num_epochs = 100
- learning_rate = 0.001
- seq_size = 30

## Data acquisition

- DAX
- YFinance API
- ECB interest rate

### Features

- X
  - From 2015 to 2022
  - Interval: daily
  - Open
  - High
  - Low
  - Close
  - Adj Close
  - Volume
  - Month
  - Weekday
  - Trend (trend is positive if open < close)
- Y
  - Buy or Sell

## Target

Higher cumulative return by our Strategy than the DAX itself.

## Performance criteria

For our Tests we use the Model from Epoch: XXXX as it was the first one with the best performance. Keep in mind there are only two performance types of our Model: always invest and always sell.

### Backtesting

- Using Library: lumibot.strategies.strategy and YFinance API
- Data from 2022 to 2023
- Interval: daily
- DAX cumulative return over time: 15.6%
- Steategy cumulative return over time:

### General Results

## Conclusion
