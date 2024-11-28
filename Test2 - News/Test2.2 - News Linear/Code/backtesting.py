from datetime import datetime, time, timedelta

import torch
from lumibot.strategies.strategy import Strategy
from data_collector import collect_data_inkl_news, determine_trend, sentiment_int
from timedelta import Timedelta
import pandas as pd

class Backtest(Strategy): 
    
    def initialize(self, dataset: pd.DataFrame, model, symbol:str="^GDAXI", cash_at_risk:float=.5, num_prior_days:int=5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.model = model
        self.num_prior_days = num_prior_days
        self.dataset = dataset

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
        
        finance_data = self.dataset[self.dataset["Date"] == today]
        finance_data = finance_data.iloc[:, 1:-1]
        finance_data["news_probability"] = finance_data["news_probability"].apply(lambda x: x.removeprefix("tensor(").removesuffix(", grad_fn=<SelectBackward0>)"))
        
        return finance_data
    
    def get_model_prediction(self): 
        data = self.get_data()
        
        if data.empty:
            raise ValueError("Finance data is empty. Check data source or date range.")
                

        data["month"] = data["month"].astype(int)
        data["news_probability"] = data["news_probability"].astype(float)        

        # Make prediction
        prediction = int(self.model(torch.tensor(data.values, dtype=torch.float32)).round().item())
                
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
            
        