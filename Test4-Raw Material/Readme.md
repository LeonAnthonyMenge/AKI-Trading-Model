# Predicting DAX Index Profitability with LSTM

## Short Description

This project utilizes a Long Short-Term Memory (LSTM) neural network to predict the profitability of buying the DAX Index on a given day. To enhance prediction accuracy, correlations with Gold and Brent Crude Oil are considered.

## Modeling Architecture

- **Model Type**: LSTM
- **Objective**: Predict DAX closing prices and generate a trading strategy.

## Model Parameters

- **Input Size**: 20
- **Output Size**: 1
- **Hidden Size**: Configurable (50 to 1000 in experiments)
- **Number of Layers**: Configurable (1 to 6 in experiments)
- **Dropout**: Configurable (0.1 to 0.2 in experiments)
- **Batch Size**: 1
- **Sequence Size**: 30

## Training Parameters

- **Batch Size**: 1
- **Number of Epochs**: 10 (Break after 8 iterations without getting better)
- **Learning Rate**: 0.0001

## Data Acquisition

- **Sources**:
  - **DAX data**: YFinance API
  - **Gold and Brent Crude Oil**: YFinance API
- **Features**:
  - **X** (Input):
    - Data range: 2015 to 2022
    - Interval: Daily
    - Columns: Open, High, Low, Close, Adjusted Close, Volume (DAX, Gold, Oil), Month, Weekday
  - **Y** (Output): Predicted closing price

## Target

The model aims to achieve higher cumulative returns using the strategy compared to the DAX index.

## Performance Criteria

- **Test Loss**: Minimum loss achieved.
- **Cumulative Return**: Total return from the trading strategy.

### Backtesting

- **Tools**: Lumibot strategies and YFinance API
- **Data**: 2022 to 2023 (daily interval)
- **Performance Comparison**:
  - **DAX Cumulative Return**: **15.6%**

## Experimental Results

| Run  | Hidden Size | Layers | Dropout | Test Loss | Strategy Time (%) | Strategy Return (%) | CAGR (%) | Trades                                                                 | Tearsheet                                                                 |
|------|-------------|--------|---------|-----------|--------------------|---------------------|----------|------------------------------------------------------------------------|---------------------------------------------------------------------------|
| T1   | 1000        | 1      | 0.2     | 0.0006    | 22.0              | 6.10                | 6.10     | [T1_trades.html](results/T1_trades.html)            | [T1_tearsheet.html](results/T1_tearsheet.html)         |
| T2   | 1000        | 1      | 0.1     | 0.0005    | 24.0              | 6.45                | 6.54     | [T2_trades.html](results/T2_trades.html)            | [T2_tearsheet.html](results/T2_tearsheet.html)         |
| T3   | 500         | 2      | 0.1     | 0.0006    | 25.0              | 11.00               | 11.17    | [T3_trades.html](results/T3_trades.html)            | [T3_tearsheet.html](results/T3_tearsheet.html)         |
| T4   | 250         | 2      | 0.1     | 0.0007    | 22.0              | 8.02                | 8.14     | [T4_trades.html](results/T4_trades.html)            | [T4_tearsheet.html](results/T4_tearsheet.html)         |
| T5   | 250         | 6      | 0.1     | 0.0019    | 25.0              | 10.72               | 10.88    | [T5_trades.html](results/T5_trades.html)            | [T5_tearsheet.html](results/T5_tearsheet.html)         |
| T6   | 500         | 4      | 0.1     | 0.0008    | 26.0              | 9.90                | 10.04    | [T6_trades.html](results/T6_trades.html)            | [T6_tearsheet.html](results/T6_tearsheet.html)         |
| T7   | 200         | 2      | 0.1     | 0.0007    | 22.0              | 9.06                | 9.19     | [T7_trades.html](results/T7_trades.html)            | [T7_tearsheet.html](results/T7_tearsheet.html)         |
| T8   | 100         | 2      | 0.1     | 0.0008    | 22.0              | 9.22                | 9.35     | [T8_trades.html](results/T8_trades.html)            | [T8_tearsheet.html](results/T8_tearsheet.html)         |
| T9   | 600         | 3      | 0.2     | 0.0007    | 21.0              | 7.97                | 7.97     | [T9_trades.html](results/T9_trades.html)            | [T9_tearsheet.html](results/T9_tearsheet.html)         |
| T10  | 100         | 1      | 0.1     | 0.0006    | 25.0              | 8.04                | 8.16     | [T10_trades.html](results/T10_trades.html)          | [T10_tearsheet.html](results/T10_tearsheet.html)       |
| T11  | 50          | 2      | 0.1     | 0.0009    | 27.0              | 10.48               | 10.64    | [T11_trades.html](results/T11_trades.html)          | [T11_tearsheet.html](results/T11_tearsheet.html)       |

## Conclusion

- The results reveal an interesting phenomenon: simpler models with higher test losses can sometimes outperform more complex models with lower losses in terms of returns. This is partly due to the positive returns in both the training and testing periods, which can favor simpler strategies.

- **Low test loss does not guarantee high returns**, and conversely, high returns can sometimes come from models with higher losses.

- **Key takeaway**: The LSTM's divergence between loss and return may suggest it isn't capturing more complex patterns, but rather exploiting the momentum present in the dataset.

- Future directions should include using different evaluation periods with varying market conditions to assess robustness.