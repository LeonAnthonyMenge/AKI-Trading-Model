from datetime import time

import torch
from lumibot.strategies.strategy import Strategy
from data_collector import get_finance_data, determine_trend
from timedelta import Timedelta

class Backtest(Strategy): 
    
    def initialize(self, model, symbol:str="DAX", cash_at_risk:float=.5, num_prior_days:int=5): 
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
    
    def get_model_prediction(self): 
        today, prior_date = self.get_dates()
        finance_data = get_finance_data(self.symbol, start=prior_date, end=today, interval="1d")
        
        if finance_data.empty:
            raise ValueError("Finance data is empty. Check data source or date range.")
        
        finance_data['trend'] = finance_data.apply(determine_trend, axis=1)
        data = finance_data.iloc[:, 1:].sort_index(ascending=False)[:self.num_prior_days]
        data["month"] = data["month"].astype(int)
        
        # Reshape with necessary adjustments
        x = data.to_numpy().reshape(1, self.num_prior_days, len(data.columns))
        print(f"X-Shape: {x.shape}")

        # Make prediction
        prediction = int(self.model(torch.tensor(x, dtype=torch.float32)).round().item())
        print(f"Prediction: {prediction}")
                
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
                        take_profit_price=last_price*1.20, 
                        stop_loss_price=last_price*.95
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
                        take_profit_price=last_price*.8, 
                        stop_loss_price=last_price*1.05
                    )
                self.submit_order(order) 
                self.last_trade = "sell"
            
        