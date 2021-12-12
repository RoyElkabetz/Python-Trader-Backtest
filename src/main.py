import numpy as np
from market import Market
from broker import Broker
from trader import Trader

# init players
tickers = ['AAPL', 'Goog', 'SPY', 'TsLa', 'Orcl']
market1 = Market(tickers, start_date=(2019, 1, 1), end_date=(2020, 1, 1))
broker1 = Broker(buy_fee=0.08, min_buy_fee=1, sell_fee=0.08, min_sell_fee=1, tax=0.25, my_market=market1)
trader1 = Trader(liquid=100000, balance_period=60, my_broker=broker1, my_market=market1)

# buy some stocks
trader1.buy('AAPL', 4)
trader1.buy('AAPL', 1)
trader1.buy('Goog', 2)
trader1.buy('Tsla', 1)

p_date = market1.step()
trader1.step(p_date)
trader1.buy('Goog', 2)
trader1.buy('Orcl', 1)

p_date = market1.step()
trader1.step(p_date)

print(trader1.portfolio_state)



