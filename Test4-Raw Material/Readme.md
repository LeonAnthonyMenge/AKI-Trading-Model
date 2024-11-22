# Short description: ​
Using a LSTM for predicting the DAX Index wether it is profitable to buy on a given day or not.
It uses Gold & Brent Crude Oil to look for possible correlations in order to improve the prediction.

##  Modeling architecture
- LSTM

### Model parameter
- input_size = 20
- output_size = 1
- hidden_size = 1000
- num_layers = 1
- dropout = 0.2

### Training parameter
- batch_size = 1
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
    - Open Dax, Gold, Oil
    - High Dax, Gold, Oil
    - Low Dax, Gold, Oil
    - Close Dax, Gold, Oil
    - Adj Close Dax, Gold, Oil
    - Volume Dax, Gold, Oil
    - Month Dax
    - Weekday Dax
- Y
    - Predicted closing

## Target
Higher cumulative return by our Strategy than the DAX itself.

## Performance criteria
Model with the lowest test loss

### Backtesting
- Using Library: lumibot.strategies.strategy and YFinance API
- Data from 2022 to 2023
- Interval: daily
- DAX cumulative return over time: 15.6%

### General Results
***siehe doku***

## Conclusion
***siehe doku***
