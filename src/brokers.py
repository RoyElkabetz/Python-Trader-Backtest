import numpy as np
import pandas as pd
import logging
from .markets import Market
from .exceptions import InvalidParameterError
from .position import Position

logger = logging.getLogger(__name__)


class Broker:
    """A Broker class which mitigate between the Trader and the Market. It execute the trades and collect the fees"""
    def __init__(self, buy_fee: float, min_buy_fee: float,
                 sell_fee: float, min_sell_fee: float, tax: float, my_market: Market):
        # Validate parameters
        if buy_fee < 0 or sell_fee < 0:
            raise InvalidParameterError("Fees cannot be negative")
        if tax < 0 or tax > 100:
            raise InvalidParameterError("Tax must be between 0 and 100 percent")
        if min_buy_fee < 0 or min_sell_fee < 0:
            raise InvalidParameterError("Minimum fees cannot be negative")
        
        self.my_market = my_market
        self.buy_fee = buy_fee / 100.
        self.min_buy_fee = min_buy_fee
        self.sell_fee = sell_fee / 100.
        self.min_sell_fee = min_sell_fee
        self.tax = tax / 100.
        
        logger.info(f"Broker initialized with buy_fee={buy_fee}%, sell_fee={sell_fee}%, tax={tax}%")

    def buy_now(self, ticker, units):
        """
        Immediate buying execution
        :param ticker: the ticker of the stock (type: str)
        :param units: the amount of units to buy (type: int)
        :return: position (type: Position), total_price (type: float), fee (type: float)
        """
        # check stock price
        price = self.my_market.get_stock_data(ticker, 'Open')
        total_price = price * units

        # Create a Position object instead of storing DataFrames
        position = Position(
            ticker=ticker,
            units=units,
            purchase_price=price,
            purchase_date=self.my_market.current_date,
            current_price=price
        )

        # compute the buying fee
        fee = self.buy_fee * total_price
        if fee < self.min_buy_fee:
            fee = self.min_buy_fee

        logger.debug(f"Buy executed: {ticker} x{units} @ ${price:.2f}, fee=${fee:.2f}")
        return position, total_price, fee

    def sell_now(self, ticker, positions):
        """
        Immediate selling execution
        :param ticker: the ticker of the stock (type: str)
        :param positions: list of Position objects to sell (type: list[Position])
        :return: current_total_price (type: float), fee (type: float), tax (type: float)
        """
        # check stock price
        current_price = self.my_market.get_stock_data(ticker, 'Open')
        
        # Calculate total units and cost basis
        total_units = sum(pos.units for pos in positions)
        current_total_price = current_price * total_units

        # compute the positions' original value (cost basis)
        positions_cost_basis = sum(pos.cost_basis for pos in positions)

        # compute the sell fee with respect to current stock price and tax
        fee = current_total_price * self.sell_fee
        if fee < self.min_sell_fee:
            fee = self.min_sell_fee
        tax = max(0, (current_total_price - positions_cost_basis) * self.tax)

        logger.debug(f"Sell executed: {ticker} x{total_units} @ ${current_price:.2f}, fee=${fee:.2f}, tax=${tax:.2f}")
        return current_total_price, fee, tax

