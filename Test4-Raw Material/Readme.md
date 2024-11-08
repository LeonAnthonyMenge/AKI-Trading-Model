# Short description: ​
Using a LSTM for predicting the DAX Index wether it is profitable to buy on a given day or not.
It uses Gold & Brent Crude Oil to look for possible correlations in order to improve the prediction.

##  Modeling architecture
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
- DAX, Gold, Brent Crude Oil
- YFinance API

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
For our Tests we use the Model from Epoch: 18 as it was the first one with the best performance. Keep in mind there are only two performance types of our Model: always invest and always sell.

### Backtesting
- Using Library: lumibot.strategies.strategy and YFinance API
- Data from 2022 to 2023
- Interval: daily
- DAX cumulative return over time: 15.6%
- Steategy cumulative return over time: 4.42%

![alt backtesting_results](results/backtesting_results.png)

### General Results
- Accuracy: 0.55
- Precision: 0.55
- Recall: 1.00
- F1 Score: 0.71

![alt text](results/confusion_matrix.png)

![alt text](results/corr_matrix.png)

## Conclusion
The model is not very accurate, every prediction of the model is to invest. Therefore it will underperform or overperform the DAX depending on its changes in trend over time.