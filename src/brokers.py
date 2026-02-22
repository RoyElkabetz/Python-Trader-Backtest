import numpy as np
import pandas as pd
from typing import Tuple, List
from .markets import Market
from .exceptions import InvalidParameterError
from .position import Position
from .logging_config import get_logger

logger = get_logger('brokers')


class Broker:
    """
    A Broker class that mediates between the Trader and the Market.
    
    The Broker executes trades (buy and sell operations), calculates transaction fees,
    and computes capital gains taxes. It enforces minimum fee requirements and validates
    all parameters to ensure proper trading operations.
    
    Parameters
    ----------
    buy_fee_percent : float
        Buy transaction fee as a percentage (e.g., 0.08 for 0.08%). Must be non-negative.
        This value is automatically converted to decimal form (divided by 100).
    min_buy_fee : float
        Minimum buy transaction fee in currency units. Applied when the percentage-based
        fee is below this threshold. Must be non-negative.
    sell_fee_percent : float
        Sell transaction fee as a percentage (e.g., 0.08 for 0.08%). Must be non-negative.
        This value is automatically converted to decimal form (divided by 100).
    min_sell_fee : float
        Minimum sell transaction fee in currency units. Applied when the percentage-based
        fee is below this threshold. Must be non-negative.
    tax : float
        Capital gains tax rate as a percentage (e.g., 25.0 for 25%). Must be between 0 and 100.
        This value is automatically converted to decimal form (divided by 100).
        Tax is only applied on profits (current price - cost basis).
    market : Market
        Market object that provides current stock prices and market data.
    
    Attributes
    ----------
    market : Market
        Reference to the Market object for price lookups.
    buy_fee_percent : float
        Buy fee as a decimal (e.g., 0.0008 for 0.08%).
    min_buy_fee : float
        Minimum buy fee in currency units.
    sell_fee_percent : float
        Sell fee as a decimal (e.g., 0.0008 for 0.08%).
    min_sell_fee : float
        Minimum sell fee in currency units.
    tax : float
        Tax rate as a decimal (e.g., 0.25 for 25%).
    
    Raises
    ------
    InvalidParameterError
        If buy_fee_percent or sell_fee_percent is negative.
        If tax is not between 0 and 100.
        If min_buy_fee or min_sell_fee is negative.
    
    Examples
    --------
    >>> from src.markets import Market
    >>> market = Market(['AAPL', 'GOOGL'], start_date=[2020, 1, 1], end_date=[2021, 1, 1])
    >>> broker = Broker(
    ...     buy_fee_percent=0.08,
    ...     min_buy_fee=2.0,
    ...     sell_fee_percent=0.08,
    ...     min_sell_fee=2.0,
    ...     tax=25.0,
    ...     market=market
    ... )
    """
    def __init__(self, buy_fee_percent: float, min_buy_fee: float,
                 sell_fee_percent: float, min_sell_fee: float, tax: float, market: Market) -> None:
        # Validate parameters
        if buy_fee_percent < 0 or sell_fee_percent < 0:
            raise InvalidParameterError("Fees cannot be negative")
        if tax < 0 or tax > 100:
            raise InvalidParameterError("Tax must be between 0 and 100 percent")
        if min_buy_fee < 0 or min_sell_fee < 0:
            raise InvalidParameterError("Minimum fees cannot be negative")
        
        self.market = market
        self.buy_fee_percent = buy_fee_percent / 100.
        self.min_buy_fee = min_buy_fee
        self.sell_fee_percent = sell_fee_percent / 100.
        self.min_sell_fee = min_sell_fee
        self.tax = tax / 100.
        
        logger.info(f"Broker initialized: buy_fee={buy_fee_percent}% (min ${min_buy_fee}), "
                   f"sell_fee={sell_fee_percent}% (min ${min_sell_fee}), tax={tax}%")

    def buy_now(self, ticker: str, units: int) -> Tuple[Position, float, float]:
        """
        Immediate buying execution
        
        Args:
            ticker: The ticker of the stock
            units: The amount of units to buy
            
        Returns:
            Tuple of (position, total_price, fee)
        """
        # check stock price
        price = self.market.get_stock_data(ticker, 'Open')
        total_price = price * units

        # Create a Position object instead of storing DataFrames
        position = Position(
            ticker=ticker,
            units=units,
            purchase_price=price,
            purchase_date=self.market.current_date,
            current_price=price
        )

        # compute the buying fee
        fee = self.buy_fee_percent * total_price
        if fee < self.min_buy_fee:
            fee = self.min_buy_fee

        logger.debug(f"Buy executed: {ticker} x{units} @ ${price:.2f}, total=${total_price:.2f}, fee=${fee:.2f}")
        return position, total_price, fee

    def sell_now(self, ticker: str, positions: List[Position]) -> Tuple[float, float, float]:
        """
        Immediate selling execution
        
        Args:
            ticker: The ticker of the stock
            positions: List of Position objects to sell
            
        Returns:
            Tuple of (current_total_price, fee, tax)
        """
        # check stock price
        current_price = self.market.get_stock_data(ticker, 'Open')
        
        # Calculate total units and cost basis
        total_units = sum(pos.units for pos in positions)
        current_total_price = current_price * total_units

        # compute the positions' original value (cost basis)
        positions_cost_basis = sum(pos.cost_basis for pos in positions)

        # compute the sell fee with respect to current stock price and tax
        fee = current_total_price * self.sell_fee_percent
        if fee < self.min_sell_fee:
            fee = self.min_sell_fee
        tax = max(0, (current_total_price - positions_cost_basis) * self.tax)

        profit = current_total_price - positions_cost_basis
        logger.debug(f"Sell executed: {ticker} x{total_units} @ ${current_price:.2f}, "
                    f"proceeds=${current_total_price:.2f}, profit=${profit:.2f}, fee=${fee:.2f}, tax=${tax:.2f}")
        return current_total_price, fee, tax

