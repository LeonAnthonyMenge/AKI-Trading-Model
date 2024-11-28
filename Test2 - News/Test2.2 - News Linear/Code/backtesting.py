import numpy as np
import torch
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta
import pandas as pd

class Backtest(Strategy): 
    
    def initialize(self, model, scaler, scaler_y, dataset: pd.DataFrame, symbol: str = "^GDAXI", cash_at_risk: float = 0.5, num_prior_days: int = 5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.model = model
        self.num_prior_days = num_prior_days
        self.dataset = dataset
        self.scaler = scaler
        self.scaler_y = scaler_y

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
        finance_data = self.scaler.transform(finance_data.values)      
        return finance_data
    
    def get_model_prediction(self): 
        data = self.get_data()      
        # Make prediction
        prediction = self.model(torch.tensor(data, dtype=torch.float32)).detach().numpy()
        prediction = self.scaler_y.inverse_transform(np.concatenate(prediction).reshape(1, -1))[0][0]
        return prediction

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred = self.get_model_prediction()
        if pred > last_price:
            if cash > last_price:
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    time_in_force="gtc",
                    type="market",
                )
                self.submit_order(order)
                self.last_trade = "buy"
        else:
            if self.last_trade == "buy":
                self.sell_all()
                self.last_trade = "sell"
            
        