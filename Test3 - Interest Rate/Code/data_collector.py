import pandas as pd
import yfinance as yf

def get_finance_data(symbols: list, period: str = "max", start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    # Fetch data
    df = yf.download(tickers=symbols, start=start, end=end, period=period, interval=interval)
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # Add Month and Day information
    df["month"] = df['Date'].dt.strftime('%m').astype(int)
    df['weekday'] = df['Date'].dt.dayofweek.astype(int)
    
    # return data
    return df


def get_ecb_interest_rates(start_date, end_date) -> pd.DataFrame:
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
