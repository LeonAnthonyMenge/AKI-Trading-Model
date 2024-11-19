from datetime import time
import torch
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta
import pandas as pd
import numpy as np

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
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        prior_date = today - Timedelta(days=(self.num_prior_days + 50))
        return today.strftime('%Y-%m-%d'), prior_date.strftime('%Y-%m-%d')

    def get_data(self) -> pd.DataFrame:
        today, prior_date = self.get_dates()
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        finance_data = self.dataset[(self.dataset["Date"] <= today) & (self.dataset["Date"] >= prior_date)]
        finance_data.sort_values("Date", inplace=True)
        finance_data.reset_index(inplace=True, drop=True)
        finance_data = finance_data.iloc[:self.num_prior_days, :]
        finance_data = finance_data[["Open", "High", "Low", "Close", "Adj Close", "Volume", "month", "weekday"]]
        finance_data = self.scaler.transform(finance_data.values).reshape(1, self.num_prior_days, len(finance_data.columns))
        return finance_data

    def get_model_prediction(self):
        today, prior_date = self.get_dates()
        self.dataset["Date"] = pd.to_datetime(self.dataset["Date"])
        finance_data = self.dataset[(self.dataset["Date"] < today) & (self.dataset["Date"] >= prior_date)]
        finance_data.sort_values("Date", inplace=True)
        finance_data.reset_index(inplace=True, drop=True)
        finance_data = finance_data.iloc[:self.num_prior_days, 2:-1]
        finance_data = self.scaler.transform(finance_data.values).reshape(1, self.num_prior_days, len(finance_data.columns))
        prediction = self.model(torch.tensor(finance_data, dtype=torch.float32)).detach().numpy()
        prediction = self.scaler_y.inverse_transform(np.concatenate(prediction).reshape(1, -1))[0][0]
        return prediction

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred = self.get_model_prediction()

        # Entscheiden basierend auf der Vorhersage
        print(f"Prediction: {pred}, Last Price: {last_price}")
        if pred > last_price:
            if cash > last_price:
                print("Buying...")
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
                print("Selling...")
                self.sell_all()
                self.last_trade = "sell"
