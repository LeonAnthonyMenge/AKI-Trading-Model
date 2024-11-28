from datetime import time
import torch
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta
import pandas as pd
import numpy as np

class Backtest(Strategy):
    def initialize(self, model, scaler_com, scaler_fg, scaler_news, scaler_y, dataset_com: pd.DataFrame, dataset_fg: pd.DataFrame, dataset_news: pd.DataFrame, symbol: str = "^GDAXI", cash_at_risk: float = 0.5, num_prior_days: int = 5, stop_loss: float = .95, take_profit:float = 1.2):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.model = model
        self.num_prior_days = num_prior_days
        self.dataset_com = dataset_com
        self.dataset_news = dataset_news
        self.dataset_fg = dataset_fg
        self.scaler_com = scaler_com
        self.scaler_fg = scaler_fg
        self.scaler_news = scaler_news
        self.scaler_y = scaler_y
        self.stop_loss = stop_loss
        self.take_profit = take_profit

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
        # Commodities
        self.dataset_com["Date"] = pd.to_datetime(self.dataset_com["Date"])
        finance_data = self.dataset_com[(self.dataset_com["Date"] <= today) & (self.dataset_com["Date"] >= prior_date)]
        finance_data.sort_values("Date", inplace=True)
        finance_data = finance_data.iloc[:self.num_prior_days, 1:-1]
        finance_data_com = self.scaler_com.transform(finance_data.values).reshape(1, self.num_prior_days, len(finance_data.columns))
         # Fear and Greed
        self.dataset_fg["Date"] = pd.to_datetime(self.dataset_fg["Date"])
        finance_data = self.dataset_fg[(self.dataset_fg["Date"] <= today) & (self.dataset_fg["Date"] >= prior_date)]
        finance_data.sort_values("Date", inplace=True)
        finance_data = finance_data.iloc[:self.num_prior_days, 1:-1]
        finance_data_fg = self.scaler_fg.transform(finance_data.values).reshape(1, self.num_prior_days, len(finance_data.columns))
         # News
        self.dataset_news["Date"] = pd.to_datetime(self.dataset_news["Date"])
        finance_data = self.dataset_news[(self.dataset_news["Date"] <= today) & (self.dataset_news["Date"] >= prior_date)]
        finance_data.sort_values("Date", inplace=True)
        finance_data = finance_data.iloc[:self.num_prior_days, 1:-1]
        finance_data_news = self.scaler_news.transform(finance_data.values).reshape(1, self.num_prior_days, len(finance_data.columns))
        return finance_data_com, finance_data_fg, finance_data_news

    def get_model_prediction(self):
        finance_data_com, finance_data_fg, finance_data_news = self.get_data()
        prediction, _ = self.model(
            torch.tensor(finance_data_com, dtype=torch.float32),
            torch.tensor(finance_data_fg, dtype=torch.float32),
            torch.tensor(finance_data_news, dtype=torch.float32),
        )
        prediction = prediction.detach().numpy()
        prediction = self.scaler_y.inverse_transform(np.concatenate(prediction).reshape(1, -1))[0][0]
        return prediction

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        pred = self.get_model_prediction()
        if pred*1.05 > last_price:
            if cash > last_price:
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    time_in_force="gtc",
                    type="market",
                    stop_loss_price=last_price*self.stop_loss,
                    take_profit_price=last_price*self.take_profit
                )
                self.submit_order(order)
                self.last_trade = "buy"
        else:
            if self.last_trade == "buy":
                self.sell_all()
                self.last_trade = "sell"
        if pred*1.15 < last_price: # careful short
            cash, last_price, quantity = self.position_sizing()
            if cash > last_price:
                order = self.create_order(
                    self.symbol,
                    quantity, 
                    "sell",
                    time_in_force="gtc",
                    type="market",
                    stop_loss_price=last_price*(2-self.stop_loss),
                    take_profit_price=last_price*(2-self.take_profit),
                )
                self.submit_order(order)
                self.last_trade = "sell_short"
