"""
Tests for the Broker class.
"""

import pytest
from datetime import date
from src.brokers import Broker
from src.markets import Market
from src.position import Position
from src.exceptions import InvalidParameterError


class TestBroker:
    """Test suite for Broker class."""
    
    @pytest.fixture
    def market(self):
        """Create a test market."""
        return Market(['AAPL'], (2020, 1, 1), (2020, 3, 31))
    
    @pytest.fixture
    def broker(self, market):
        """Create a test broker."""
        return Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
    
    def test_broker_creation(self, market):
        """Test creating a Broker object."""
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        assert broker.buy_fee == 0.001
        assert broker.min_buy_fee == 1.0
        assert broker.sell_fee == 0.001
        assert broker.min_sell_fee == 1.0
        assert broker.tax == 0.25
    
    def test_negative_fee_raises_error(self, market):
        """Test that negative fees raise InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            Broker(-0.1, 1.0, 0.1, 1.0, 25.0, market)
    
    def test_invalid_tax_raises_error(self, market):
        """Test that invalid tax raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError):
            Broker(0.1, 1.0, 0.1, 1.0, 150.0, market)
    
    def test_buy_now_returns_position(self, broker, market):
        """Test that buy_now returns a Position object."""
        market.step()
        position, total_price, fee = broker.buy_now('AAPL', 10)
        
        assert isinstance(position, Position)
        assert position.ticker == 'AAPL'
        assert position.units == 10
        assert total_price > 0
        assert fee >= broker.min_buy_fee
    
    def test_buy_now_minimum_fee(self, broker, market):
        """Test that minimum buy fee is applied."""
        market.step()
        position, total_price, fee = broker.buy_now('AAPL', 1)
        assert fee == broker.min_buy_fee
    
    def test_sell_now_returns_values(self, broker, market):
        """Test that sell_now returns correct values."""
        market.step()
        position, _, _ = broker.buy_now('AAPL', 10)
        
        # Sell the position
        money, fee, tax = broker.sell_now('AAPL', [position])
        
        assert money > 0
        assert fee >= broker.min_sell_fee
        assert tax >= 0
    
    def test_sell_now_calculates_tax_on_profit(self, broker, market):
        """Test that tax is calculated on profit."""
        market.step()
        # Get actual market price
        current_price = market.get_stock_data('AAPL', 'Open')
        position = Position('AAPL', 10, 100.0, market.current_date, current_price=current_price)
        
        money, fee, tax = broker.sell_now('AAPL', [position])
        
        # Tax should be on the profit if current price > purchase price
        if current_price > 100.0:
            expected_profit = (current_price * 10 - 1000)
            expected_tax = expected_profit * 0.25
            assert tax > 0
            assert abs(tax - expected_tax) < 0.01
        else:
            assert tax == 0
    
    def test_sell_now_no_tax_on_loss(self, broker, market):
        """Test that no tax is charged on losses."""
        market.step()
        position = Position('AAPL', 10, 100.0, market.current_date, current_price=90.0)
        
        money, fee, tax = broker.sell_now('AAPL', [position])
        
        assert tax == 0

# Made with Bob
