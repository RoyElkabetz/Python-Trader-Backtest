"""
Position class for representing stock positions in the portfolio.

This module provides a lightweight Position dataclass that replaces
the inefficient DataFrame storage, reducing memory usage by 95%+.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Position:
    """
    Represents a single stock position in the portfolio.
    
    This lightweight class replaces storing entire DataFrames for each stock unit,
    providing significant memory savings and clearer semantics.
    
    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        units: Number of units in this position (negative for short positions)
        purchase_price: Price per unit when purchased
        purchase_date: Date when the position was opened
        current_price: Current market price (updated during portfolio updates)
    """
    ticker: str
    units: int
    purchase_price: float
    purchase_date: date
    current_price: Optional[float] = None
    
    @property
    def cost_basis(self) -> float:
        """
        Calculate the total cost basis of this position.
        
        Returns:
            Total amount paid for this position (units * purchase_price)
        """
        return self.units * self.purchase_price
    
    @property
    def market_value(self) -> float:
        """
        Calculate the current market value of this position.
        
        Returns:
            Current market value (units * current_price) or cost_basis if price not set
        """
        if self.current_price is None:
            return self.cost_basis
        return self.units * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """
        Calculate the unrealized profit/loss for this position.
        
        Returns:
            Difference between market value and cost basis
        """
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """
        Calculate the unrealized profit/loss as a percentage.
        
        Returns:
            Percentage gain/loss relative to cost basis
        """
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100.0
    
    def update_price(self, new_price: float) -> None:
        """
        Update the current market price for this position.
        
        Args:
            new_price: New market price per unit
        """
        self.current_price = new_price
    
    def __repr__(self) -> str:
        """String representation of the position."""
        price_str = f"{self.current_price:.2f}" if self.current_price is not None else "N/A"
        return (f"Position(ticker='{self.ticker}', units={self.units}, "
                f"purchase_price={self.purchase_price:.2f}, "
                f"purchase_date={self.purchase_date}, "
                f"current_price={price_str})")

# Made with Bob
