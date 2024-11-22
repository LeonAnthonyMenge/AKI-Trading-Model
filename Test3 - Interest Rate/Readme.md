# Short description: 

Using a LSTM for predicting the DAX Index whether it is profitable to buy on a given day or not. To improve accuracy the current ecb interest rate was added as data.

## Modeling architecture

- LSTM

### Model parameter

- input_size = 7
- output_size = 1
- hidden_size = 1000
- num_layers = 1
- dropout = 0.2
- batch_size = 1

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
- Y
  - Predicted closing price

## Target

Higher cumulative return by our Strategy than the DAX itself.

## Performance criteria

Model with the lowest test-loss.

### Backtesting

- Using Library: lumibot.strategies.strategy and YFinance API
- Data from 2022 to 2023
- Interval: daily
- DAX cumulative return over time: 15.6%
- Steategy cumulative return over time:

### General Results

## Conclusion
