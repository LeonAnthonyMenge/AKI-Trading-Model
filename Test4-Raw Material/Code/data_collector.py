import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import torch

load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
        
def get_finance_data(symbols: list, period: str = "max", start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Args:
        symbols (list): 
            List of Symbols, whitch we want to get data for
        period (str, optional): 
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end. Defaults to "max".
        start (str, optional): 
            start date in YYYY-MM-DD format. 
            Defaults to None.
        end (str, optional): 
            end date in YYYY-MM-DD format. 
            Defaults to None.
        interval (str, optional): 
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days. 
            Defaults to "1d".

    Returns:
        pd.DataFrame: Contains the optimized finance data for the given symbols grouped by column
    """
    # Fetch data
    df = yf.download(tickers=symbols, start=start, end=end, period=period, interval=interval)
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # Add Month and Day information
    df["month"] = df['Date'].dt.strftime('%m').astype(int)
    df['weekday'] = df['Date'].dt.dayofweek.astype(int)
    
    # return data
    return df

def determine_trend(row) -> int:
    """
    Summary:
        Determine if the Close is higher than the Open (last day Close)
    Args:
        row (_type_): Pandas Dataframe row

    Returns:
        int: -1 if Trend is going down and 1 if Trend is going up
    """
    print(type(row))
    return -1 if row['Open'] > row['Close'] else 1