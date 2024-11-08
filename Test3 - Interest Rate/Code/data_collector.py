import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]
        
def get_finance_data(symbols: list, period: str = "max", start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Args:
        symbols (list): 
            List of Symbols, witch we want to get data for
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


def get_ecb_interest_rates(start_date: str, end_date: str) -> pd.DataFrame:
    data_path = "../Data/ECB_Data_Raw.csv"

    df = pd.read_csv(data_path, parse_dates=["DATE"], usecols=["DATE",
                                                               "Main refinancing operations - fixed rate tenders (fixed rate) (date of changes) - Level (FM.B.U2.EUR.4F.KR.MRR_FR.LEV)"])

    df.columns = ["DATE", "INTEREST_RATE"]

    df.set_index("DATE", inplace=True)
    df = df.resample('D').ffill()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    completed_df = df.reindex(date_range).ffill().reset_index()
    completed_df.columns = ["DATE", "INTEREST_RATE"]

    return completed_df
