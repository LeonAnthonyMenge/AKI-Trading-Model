from datetime import time

import torch
from lumibot.strategies.strategy import Strategy
from data_collector import get_finance_data, determine_trend
from timedelta import Timedelta
import pandas as pd

class Backtest(Strategy): 
    
    def initialize(self, model, scaler, dataset: pd.DataFrame, symbol:str="^GDAXI", cash_at_risk:float=.5, num_prior_days:int=5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.model = model
        self.num_prior_days = num_prior_days
        self.dataset = dataset
        self.scaler = scaler

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        prior_date = today - Timedelta(days=(self.num_prior_days+20))
        return today.strftime('%Y-%m-%d'), prior_date.strftime('%Y-%m-%d')
    
    def get_data(self) -> pd.DataFrame:
        today, prior_date = self.get_dates()
        
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        self.dataset = self.dataset[['Date','^GDAXI_Open', '^GDAXI_High', '^GDAXI_Low', '^GDAXI_Close',
                                '^GDAXI_Adj Close', '^GDAXI_Volume', '^GDAXI_month', '^GDAXI_weekday',
                                'GC=F_Open', 'GC=F_High', 'GC=F_Low', 'GC=F_Close',
                                'GC=F_Adj Close', 'GC=F_Volume', 'GC=F_month', 'GC=F_weekday',
                                'BZ=F_Open', 'BZ=F_High', 'BZ=F_Low', 'BZ=F_Close',
                                'BZ=F_Adj Close', 'BZ=F_Volume', 'BZ=F_month', 'BZ=F_weekday',
                                '^GDAXI_trend', 'Invest']]
        finance_data = self.dataset[(self.dataset["Date"] <= today) & (self.dataset["Date"] >= prior_date)]
        finance_data.set_index("Date", inplace=True)
        finance_data = finance_data.sort_index(ascending=False)
        finance_data.reset_index(inplace=True)
        finance_data = finance_data.iloc[:self.num_prior_days, 1:-1]
        finance_data = self.scaler.transform(finance_data.iloc[:self.num_prior_days, :].values).reshape(1, self.num_prior_days, len(finance_data.columns))
        return finance_data
    
    def get_model_prediction(self): 
        data = self.get_data()

        # Make prediction
        prediction = int(self.model(torch.tensor(data, dtype=torch.float32)).round().item())
                
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
            
        