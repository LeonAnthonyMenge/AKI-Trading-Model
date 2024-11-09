from datetime import time

import torch
from lumibot.strategies.strategy import Strategy
from data_collector import collect_data_inkl_news, determine_trend, sentiment_int
from timedelta import Timedelta
import pandas as pd

class Backtest(Strategy): 
    
    def initialize(self, model, symbol:str="^GDAXI", cash_at_risk:float=.5, num_prior_days:int=5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.model = model
        self.num_prior_days = num_prior_days

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        prior_date = today - Timedelta(days=(self.num_prior_days+10))
        return today.strftime('%Y-%m-%d'), prior_date.strftime('%Y-%m-%d')
    
    def get_data(self) -> pd.DataFrame:
        today, prior_date = self.get_dates()
        
        finance_data = collect_data_inkl_news(self.symbol, start=prior_date, end=today, alpaca_symbol="SPY")
        # finance_data.drop(columns=["Unnamed: 0"], inplace=True)
        
        finance_data = finance_data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "month", "weekday", "news_probability", "news_sentiment"]]
        # finance_data["news_probability"] = finance_data["news_probability"].apply(lambda x: float(x.removeprefix("tensor(").split(",")[0]))
        finance_data["trend"] = finance_data.apply(determine_trend, axis=1)
        finance_data["sentiment_int"] = finance_data.apply(sentiment_int, axis=1)
        finance_data.set_index("Date", inplace=True)
        finance_data.sort_index(inplace=True)
        finance_data.reset_index(inplace=True)
        finance_data = finance_data.drop(columns=["Date", "High", "Low", "Adj Close", "Volume", "news_sentiment"])

        finance_data['Invest'] = 0
        
        for index, row in finance_data.iterrows():
            if index > 0: 
                if finance_data.at[index, "trend"] == 1:
                    finance_data.at[index-1, "Invest"] = True
                else:
                    finance_data.at[index-1, "Invest"] = False

        finance_data["month"] = finance_data["month"].astype(int)
        finance_data["weekday"] = finance_data["weekday"].astype(int)
        finance_data["Invest"] = finance_data["Invest"].astype(bool)
        
        return finance_data
    
    def get_model_prediction(self): 
        finance_data = self.get_data()
        
        if finance_data.empty:
            raise ValueError("Finance data is empty. Check data source or date range.")
        

        data = finance_data.iloc[:, :-1][:self.num_prior_days]
        

        data["month"] = data["month"].astype(int)
        data["news_probability"] = data["news_probability"].astype(float)
        
        # Reshape with necessary adjustments
        x = data.to_numpy().reshape(1, self.num_prior_days, len(data.columns))

        # Make prediction
        prediction = int(self.model(torch.tensor(x, dtype=torch.float32)).round().item())
                
        return prediction

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        if self.get_model_prediction() == 1:
            if cash > last_price: 
                order = self.create_order(
                        self.symbol, 
                        quantity, 
                        "buy", 
                        type="bracket", 
                    )
                self.submit_order(order) 
                self.last_trade = "buy"
        elif self.get_model_prediction() == 0:
            if self.last_trade == "buy":
                self.sell_all() 
            if cash > last_price: 
                order = self.create_order(
                        self.symbol, 
                        quantity, 
                        "sell", 
                        type="bracket", 
                    )
                self.submit_order(order) 
                self.last_trade = "sell"
            
        