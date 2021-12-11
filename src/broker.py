import numpy as np
import pandas as pd


class Broker:
    def __init__(self, periodic_fee: tuple, buy_fee: float,
                 sell_fee: float, time_delay: float, tax: float):
        self.periodic_fee = periodic_fee
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.time_delay = time_delay
        self.tax = tax


