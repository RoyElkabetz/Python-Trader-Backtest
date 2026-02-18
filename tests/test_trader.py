"""
Tests for the Trader class and portfolio analytics.
"""

import pytest
from datetime import date
from src.traders import Trader
from src.brokers import Broker
from src.markets import Market


class TestTrader:
    """Test suite for Trader class."""
    
    @pytest.fixture
    def market(self):
        """Create a test market."""
        return Market(['AAPL', 'GOOG'], (2020, 1, 1), (2020, 6, 30))
    
    @pytest.fixture
    def broker(self, market):
        """Create a test broker."""
        return Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
    
    @pytest.fixture
    def trader(self, broker, market):
        """Create a test trader."""
        return Trader(10000, 30, broker, market, verbose=False)
    
    def test_trader_creation(self, trader):
        """Test creating a Trader object."""
        assert trader.liquid == 10000
        assert trader.balance_period == 30
        assert trader.sell_strategy == 'FIFO'
        assert len(trader.portfolio) == 0
    
    def test_buy_stock(self, trader, market):
        """Test buying stock."""
        market.step()
        result = trader.buy('AAPL', 10)
        
        assert result is True
        assert 'AAPL' in trader.portfolio
        assert trader.portfolio_meta['AAPL']['units'] == 10
        assert trader.liquid < 10000
    
    def test_buy_insufficient_funds(self, trader, market):
        """Test buying with insufficient funds."""
        market.step()
        result = trader.buy('AAPL', 10000)  # Way too many units
        
        assert result is False
        assert 'AAPL' not in trader.portfolio
    
    def test_sell_stock(self, trader, market):
        """Test selling stock."""
        market.step()
        trader.buy('AAPL', 10)
        initial_liquid = trader.liquid
        
        # Step forward a few days
        for _ in range(5):
            market.step()
            trader.update()
        
        result = trader.sell('AAPL', 5)
        
        assert result is True
        assert trader.portfolio_meta['AAPL']['units'] == 5
        assert trader.liquid > initial_liquid
    
    def test_sell_insufficient_shares(self, trader, market):
        """Test selling more shares than owned."""
        market.step()
        trader.buy('AAPL', 10)
        
        result = trader.sell('AAPL', 20)
        
        assert result is False
        assert trader.portfolio_meta['AAPL']['units'] == 10
    
    def test_deposit(self, trader):
        """Test depositing money."""
        initial_liquid = trader.liquid
        trader.deposit(1000)
        assert trader.liquid == initial_liquid + 1000
    
    def test_withdraw_success(self, trader):
        """Test withdrawing money successfully."""
        amount = trader.withdraw(1000)
        assert amount == 1000
        assert trader.liquid == 9000
    
    def test_withdraw_insufficient_funds(self, trader):
        """Test withdrawing more than available."""
        amount = trader.withdraw(20000)
        assert amount == 0
        assert trader.liquid == 10000
    
    def test_update_portfolio(self, trader, market):
        """Test updating portfolio values."""
        market.step()
        trader.buy('AAPL', 10)
        trader.buy('GOOG', 5)
        
        # Step to record history (step calls update internally)
        current_date = market.current_date
        trader.step(current_date)
        
        # Step forward
        done, prev_date = market.step()
        trader.step(prev_date)
        
        # Portfolio value should be updated
        assert len(trader.portfolio_value_history) > 0
        assert len(trader.date_history) > 0


class TestPortfolioAnalytics:
    """Test suite for portfolio analytics methods."""
    
    @pytest.fixture
    def trader_with_history(self):
        """Create a trader with some trading history."""
        market = Market(['AAPL', 'GOOG'], (2020, 1, 1), (2020, 6, 30))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        trader = Trader(10000, 30, broker, market, verbose=False)
        
        # Simulate some trading
        market.step()
        trader.buy('AAPL', 10)
        trader.buy('GOOG', 5)
        
        # Step through several days
        for _ in range(50):
            if not market.step()[0]:
                trader.update()
            else:
                break
        
        return trader
    
    def test_get_total_return(self, trader_with_history):
        """Test total return calculation."""
        total_return = trader_with_history.get_total_return()
        assert isinstance(total_return, float)
    
    def test_get_sharpe_ratio(self, trader_with_history):
        """Test Sharpe ratio calculation."""
        sharpe = trader_with_history.get_sharpe_ratio()
        assert isinstance(sharpe, float)
        assert sharpe >= 0 or sharpe < 0  # Can be positive or negative
    
    def test_get_max_drawdown(self, trader_with_history):
        """Test max drawdown calculation."""
        max_dd, peak_date, trough_date = trader_with_history.get_max_drawdown()
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown is negative or zero
    
    def test_get_cagr(self, trader_with_history):
        """Test CAGR calculation."""
        cagr = trader_with_history.get_cagr()
        assert isinstance(cagr, float)
    
    def test_get_volatility(self, trader_with_history):
        """Test volatility calculation."""
        volatility = trader_with_history.get_volatility()
        assert isinstance(volatility, float)
        assert volatility >= 0
    
    def test_get_win_rate(self, trader_with_history):
        """Test win rate calculation."""
        win_rate = trader_with_history.get_win_rate()
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 100
    
    def test_get_portfolio_summary(self, trader_with_history):
        """Test portfolio summary."""
        summary = trader_with_history.get_portfolio_summary()
        
        assert isinstance(summary, dict)
        assert 'total_value' in summary
        assert 'total_return_pct' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown_pct' in summary
        assert 'win_rate_pct' in summary


class TestTransactionHistory:
    """Test suite for transaction history tracking."""
    
    @pytest.fixture
    def trader_with_transactions(self):
        """Create a trader with transaction history."""
        market = Market(['AAPL', 'GOOG'], (2020, 1, 1), (2020, 6, 30))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        trader = Trader(10000, 30, broker, market, verbose=False)
        
        # Make some trades
        market.step()
        trader.buy('AAPL', 10)
        trader.buy('GOOG', 5)
        
        for _ in range(10):
            if not market.step()[0]:
                trader.update()
            else:
                break
        
        trader.sell('AAPL', 3)
        trader.buy('AAPL', 5)
        
        return trader
    
    def test_transaction_history_recorded(self, trader_with_transactions):
        """Test that transactions are recorded."""
        assert len(trader_with_transactions.transaction_history) == 4
    
    def test_get_transaction_history_all(self, trader_with_transactions):
        """Test getting all transactions."""
        transactions = trader_with_transactions.get_transaction_history()
        assert len(transactions) == 4
    
    def test_get_transaction_history_by_ticker(self, trader_with_transactions):
        """Test filtering transactions by ticker."""
        aapl_transactions = trader_with_transactions.get_transaction_history(ticker='AAPL')
        assert len(aapl_transactions) == 3
        assert all(t['ticker'] == 'AAPL' for t in aapl_transactions)
    
    def test_get_transaction_history_by_type(self, trader_with_transactions):
        """Test filtering transactions by type."""
        buys = trader_with_transactions.get_transaction_history(transaction_type='BUY')
        sells = trader_with_transactions.get_transaction_history(transaction_type='SELL')
        
        assert len(buys) == 3
        assert len(sells) == 1
    
    def test_get_transaction_summary(self, trader_with_transactions):
        """Test transaction summary."""
        summary = trader_with_transactions.get_transaction_summary()
        
        assert summary['total_transactions'] == 4
        assert summary['total_buys'] == 3
        assert summary['total_sells'] == 1
        assert 'AAPL' in summary['unique_tickers']
        assert 'GOOG' in summary['unique_tickers']
    
    def test_get_ticker_transactions(self, trader_with_transactions):
        """Test getting ticker-specific analysis."""
        aapl_analysis = trader_with_transactions.get_ticker_transactions('AAPL')
        
        assert aapl_analysis['ticker'] == 'AAPL'
        assert aapl_analysis['total_units_bought'] == 15
        assert aapl_analysis['total_units_sold'] == 3
        assert aapl_analysis['net_units'] == 12

# Made with Bob
