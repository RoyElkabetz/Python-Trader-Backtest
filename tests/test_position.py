"""
Tests for the Position class.
"""

import pytest
from datetime import date
from src.position import Position


class TestPosition:
    """Test suite for Position class."""
    
    def test_position_creation(self):
        """Test creating a Position object."""
        pos = Position(
            ticker='AAPL',
            units=10,
            purchase_price=150.0,
            purchase_date=date(2020, 1, 1),
            current_price=160.0
        )
        
        assert pos.ticker == 'AAPL'
        assert pos.units == 10
        assert pos.purchase_price == 150.0
        assert pos.purchase_date == date(2020, 1, 1)
        assert pos.current_price == 160.0
    
    def test_cost_basis(self):
        """Test cost basis calculation."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1))
        assert pos.cost_basis == 1500.0
    
    def test_market_value_with_current_price(self):
        """Test market value with current price set."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1), current_price=160.0)
        assert pos.market_value == 1600.0
    
    def test_market_value_without_current_price(self):
        """Test market value defaults to cost basis when current price not set."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1))
        assert pos.market_value == 1500.0
    
    def test_unrealized_pnl_profit(self):
        """Test unrealized P&L calculation for profit."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1), current_price=160.0)
        assert pos.unrealized_pnl == 100.0
    
    def test_unrealized_pnl_loss(self):
        """Test unrealized P&L calculation for loss."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1), current_price=140.0)
        assert pos.unrealized_pnl == -100.0
    
    def test_unrealized_pnl_percent(self):
        """Test unrealized P&L percentage calculation."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1), current_price=165.0)
        assert pos.unrealized_pnl_percent == 10.0
    
    def test_update_price(self):
        """Test updating current price."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1))
        pos.update_price(170.0)
        assert pos.current_price == 170.0
        assert pos.market_value == 1700.0
    
    def test_repr(self):
        """Test string representation."""
        pos = Position('AAPL', 10, 150.0, date(2020, 1, 1), current_price=160.0)
        repr_str = repr(pos)
        assert 'AAPL' in repr_str
        assert '10' in repr_str
        assert '150.00' in repr_str

# Made with Bob
