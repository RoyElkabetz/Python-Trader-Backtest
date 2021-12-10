import numpy as np
from market import Stock, Market

stocks = Market(['Aapl', 'goog', 'spy'], (2020, 1, 1), (2021, 1, 1))
print(stocks)
print('\n')
print(stocks.get_stock_data('aapl', 'Close'))
print('\n')
print(stocks.get_stock_data('aapl', 'Close').to_numpy())
print('\n')
print(stocks.check_date_((2024, 1, 1)))
print(stocks.check_date_((2020, 1, 2)))


print('\n')
print(stocks.get_date_data((2020, 1, 2)))
print('\n')
print(stocks.get_date_data((2020, 1, 2), as_numpy=True))


