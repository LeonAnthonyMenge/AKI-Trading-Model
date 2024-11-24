
# Predicting DAX Index Profitability with LSTM

## Short Description

This project explores the use of a Long Short-Term Memory (LSTM) neural network to predict whether it is profitable to buy the DAX index on a given day. To improve model accuracy, the current European Central Bank (ECB) interest rate was included as a feature.

## Modeling Architecture

- **Model Type**: LSTM
- **Objective**: Predict DAX closing prices to generate a trading strategy.

## Model Parameters

- **Input Size**: 7
- **Output Size**: 1
- **Hidden Size**: 1 - 200
- **Number of Layers**: 1 - 3 
- **Dropout**: 0.1 - 0.4
- **Batch Size**: 1
- **Sequence Size**: 1 - 30

## Training Parameters

- **Batch Size**: 1
- **Number of Epochs**: Stops after 8 epochs without getting better
- **Learning Rate**: 0.0001

## Data Acquisition

- **Sources**:
  - **DAX data**: YFinance API
  - **ECB interest rate**: External CSV file
- **Features**:
  - **X** (Input):
    - Data range: 2015 to 2022
    - Interval: Daily
    - Columns: Open, High, Low, Close, Adjusted Close, Volume, ECB interest rate
  - **Y** (Output): Predicted closing price

## Target

The model aims to achieve higher cumulative returns using the strategy than the DAX itself.

## Performance Criteria

- Models are evaluated based on:
  - **Test Loss**: Minimum test loss achieved.
  - **Return**: Cumulative return from the trading strategy.

### Backtesting

- **Tools**: Lumibot strategies and YFinance API
- **Data**: 2022 to 2023 (daily interval)
- **Performance Comparison**:
  - **DAX cumulative return**: **15.6%**
  - **Best Loss Strategy**: **10.76% return**
  - **Best Return Strategy**: **14.59% return**

## Experimental Results

| Run  | Hidden Size | Layers | Dropout | Seq Size | Min Test Loss      | Return (%) | Trades                                      | Perfomance                                    |
|------|-------------|--------|---------|----------|--------------------|------------|---------------------------------------------|-----------------------------------------------|
| 1    | 100         | 1      | 0.2     | 30       | 0.000620222606367  | 10.76      | [Trades1.html](results%2FRun1_trades.html)  | [Perfomance1.html](results%2FRun1_tearsheet.html)   |
| 2    | 100         | 2      | 0.2     | 30       | 0.000788078277647  | 7.88       | [Trades2.html](results%2FRun1_trades.html)  | [Perfomance2.html](results%2FRun2_tearsheet.html)   |
| 3    | 100         | 3      | 0.2     | 30       | 0.000974616196069  | 7.51       | [Trades3.html](results%2FRun1_trades.html)  | [Perfomance3.html](results%2FRun3_tearsheet.html)   |
| 4    | 40          | 1      | 0.2     | 30       | 0.000641744519575  | 10.27      | [Trades4.html](results%2FRun1_trades.html)  | [Perfomance4.html](results%2FRun4_tearsheet.html)   |
| 5    | 40          | 2      | 0.2     | 30       | 0.001898406070079  | 9.56       | [Trades5.html](results%2FRun1_trades.html)  | [Perfomance5.html](results%2FRun5_tearsheet.html)   |
| 6    | 60          | 1      | 0.1     | 30       | 0.000641350149001  | 9.80       | [Trades6.html](results%2FRun1_trades.html)  | [Perfomance6.html](results%2FRun6_tearsheet.html)   |
| 7    | 40          | 1      | 0.4     | 30       | 0.000680008666106  | 10.95      | [Trades7.html](results%2FRun1_trades.html)  | [Perfomance7.html](results%2FRun7_tearsheet.html)   |
| 8    | 100         | 1      | 0.2     | 3        | 0.000810342993011  | 14.59      | [Trades8.html](results%2FRun1_trades.html)  | [Perfomance8.html](results%2FRun8_tearsheet.html)   |
| 9    | 200         | 1      | 0.2     | 10       | 0.000646841717763  | 7.52       | [Trades9.html](results%2FRun1_trades.html)  | [Perfomance9.html](results%2FRun9_tearsheet.html)   |
| 10   | 100         | 1      | 0.2     | 1        | 0.000643273457123  | 14.38      | [Trades10.html](results%2FRun1_trades.html) | [Perfomance10.html](results%2FRun10_tearsheet.html) |
| 11   | 1           | 1      | 0       | 1        | 0.005727933009098  | 14.38      | [Trades11.html](results%2FRun1_trades.html) | [Perfomance11.html](results%2FRun11_tearsheet.html) |

## Conclusion

- The results reveal an interesting phenomenon: simpler models with higher test losses can sometimes outperform more complex models with lower losses in terms of returns. This is partly due to the positive returns in both the training and testing periods, which can favor simpler strategies.

- **Low test loss does not guarantee high returns**, and conversely, high returns can sometimes come from models with higher losses.

- Models with a low sequence size (e.g., 1 or 3) tend to perform better in this experiment. However, this likely indicates that the model relies heavily on the most recent day’s data and adds minor adjustments. This behavior is likely due to historical trends for the DAX, since in both training and testing period it rose significantly.

- **Key takeaway**: The LSTM's divergence between loss and return may suggest it isn't capturing more complex patterns, but rather exploiting the momentum present in the dataset.

- Future directions should include using different evaluation periods with varying market conditions to assess robustness.
