import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import os
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


def get_news(symbol: str, start: str, end: str, limit: int = 5, include_content: bool = False) -> list:
    """_summary_

    Args:
        symbol (str): name of Index
        start (str): start date in format "%Y-%m-%d"
        end (str): end date in format "%Y-%m-%d"
        limit (int, optional): Max number of news to return. Defaults to 5.
        include_content (bool, optional): Include contets in news. Defaults to False.

    Returns:
        list: List of Tuplas with news. Each containing (headline, summary)
    """
    url = "https://data.alpaca.markets/v1beta1/news" 

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret
    }

    params = {
        "symbols": symbol,
        "start": start,
        "end": end,
        "limit": limit,
        "include_content": include_content
    }
    
    response = requests.get(url, headers=headers, params=params)
    res = response.json()
    
    if res["news"]:
        news=[]
        for new in res["news"]:
            news.append((new["headline"], new["summary"]))
    else:
        news = None
        
    return news
        
        
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

def estimate_sentiment(news: str):
    """
    Summary:
        calculates sentiment of news
    Args:
        news (str): _description_
    Returns:
        probability: how sure the model is about the sentiment
        sentiment: positive, negatetive, neutral
    """
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
            "logits"
        ]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        return 0, labels[-1]
    
    
def collect_data_inkl_news(symbol: str, start: str, end:str) -> pd.DataFrame:
    """
    Summary
        collects data from yahoo finance
        get news for finance data drom Alpaca
        and estimate sentiment for news
        
    Args:
        symbol (str): Symbo of the stock for example AAPL
        start (str): start date
        end (str): end date

    Returns:
        pd.DataFrame: dataframe with news and sentiment for finance data
    """
    df = get_finance_data(symbols=symbol, start=start, end=end)

    df['news'] = None
    df["news_probability"] = None
    df["news_sentiment"] = None

    for index, row in df.iterrows():
        # get news for day
        news = get_news(symbol, row["Date"].strftime("%Y-%m-%d"), (row["Date"] + timedelta(days=1)).strftime("%Y-%m-%d"))
        df.loc[index, 'news'] = str(news)
        # get sentiment for news
        probability, sentsiment = estimate_sentiment(str(news))
        probability = probability.to(torch.float32)
        df.loc[index, 'news_probability'] = probability
        df.loc[index, 'news_sentiment'] = sentsiment
    
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