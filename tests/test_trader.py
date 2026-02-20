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


class TestTraderMicroMethods:
    """Test suite for Trader micro-methods (internal helper methods)."""
    
    @pytest.fixture
    def market(self):
        """Create a test market."""
        return Market(['AAPL', 'GOOG'], (2020, 1, 1), (2020, 6, 30))
    
    @pytest.fixture
    def broker(self, market):
        """Create a test broker."""
        # Broker divides fees by 100, so 0.1 becomes 0.001 (0.1%)
        return Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
    
    @pytest.fixture
    def trader(self, broker, market):
        """Create a test trader."""
        return Trader(10000, 30, broker, market, verbose=False)
    
    # Tests for buy() micro-methods
    
    def test_calculate_buy_cost(self, trader):
        """Test _calculate_buy_cost returns correct total cost and fee."""
        units = 10
        price = 150.0
        
        total_cost, estimated_fee = trader._calculate_buy_cost(units, price)
        
        # Total cost should be units * price
        assert total_cost == 1500.0
        # Fee should be max of percentage fee or minimum fee
        # buy_fee = 0.1% = 0.001, min_buy_fee = 1.0
        # 1500 * 0.001 = 1.5, max(1.5, 1.0) = 1.5
        expected_fee = max(trader.broker.buy_fee * total_cost, trader.broker.min_buy_fee)
        assert estimated_fee == expected_fee
        assert estimated_fee == 1.5
    
    def test_calculate_buy_cost_high_value(self, trader):
        """Test _calculate_buy_cost with high value where percentage fee exceeds minimum."""
        units = 10000
        price = 150.0
        
        total_cost, estimated_fee = trader._calculate_buy_cost(units, price)
        
        assert total_cost == 1500000.0
        # buy_fee = 0.1% = 0.001, min_buy_fee = 1.0
        # 1,500,000 * 0.001 = 1500, max(1500, 1.0) = 1500
        expected_fee = 1500000.0 * trader.broker.buy_fee
        assert estimated_fee == expected_fee
        assert estimated_fee == 1500.0
    
    def test_validate_buy_funds_sufficient(self, trader):
        """Test _validate_buy_funds returns True when funds are sufficient."""
        result = trader._validate_buy_funds('AAPL', 5000.0, 50.0)
        assert result is True
    
    def test_validate_buy_funds_insufficient(self, trader):
        """Test _validate_buy_funds returns False when funds are insufficient."""
        result = trader._validate_buy_funds('AAPL', 9000.0, 1500.0)
        assert result is False
    
    def test_validate_buy_funds_exact(self, trader):
        """Test _validate_buy_funds with exact amount available."""
        result = trader._validate_buy_funds('AAPL', 9000.0, 1000.0)
        assert result is True
    
    def test_update_portfolio_after_buy_new_ticker(self, trader, market):
        """Test _update_portfolio_after_buy with a new ticker."""
        market.step()
        position, _, _ = trader.broker.buy_now('AAPL', 10)
        
        initial_primary_value = trader.portfolio_primary_value
        trader._update_portfolio_after_buy('AAPL', position, 10)
        
        assert 'AAPL' in trader.portfolio
        assert len(trader.portfolio['AAPL']) == 1
        assert trader.portfolio_meta['AAPL']['units'] == 10
        assert trader.portfolio_primary_value == initial_primary_value + position.cost_basis
    
    def test_update_portfolio_after_buy_existing_ticker(self, trader, market):
        """Test _update_portfolio_after_buy with an existing ticker."""
        market.step()
        
        # First buy
        position1, _, _ = trader.broker.buy_now('AAPL', 10)
        trader._update_portfolio_after_buy('AAPL', position1, 10)
        
        # Second buy
        position2, _, _ = trader.broker.buy_now('AAPL', 5)
        trader._update_portfolio_after_buy('AAPL', position2, 5)
        
        assert len(trader.portfolio['AAPL']) == 2
        assert trader.portfolio_meta['AAPL']['units'] == 15
    
    def test_log_buy_transaction(self, trader, market):
        """Test _log_buy_transaction creates correct transaction record."""
        market.step()
        price = market.get_stock_data('AAPL', 'Open')
        
        initial_count = len(trader.transaction_history)
        trader._log_buy_transaction('AAPL', 10, price, 1500.0, 25.0)
        
        assert len(trader.transaction_history) == initial_count + 1
        transaction = trader.transaction_history[-1]
        assert transaction['type'] == 'BUY'
        assert transaction['ticker'] == 'AAPL'
        assert transaction['units'] == 10
        assert transaction['price'] == price
        assert transaction['total_value'] == 1500.0
        assert transaction['fee'] == 25.0
        assert transaction['tax'] == 0
    
    # Tests for sell() micro-methods
    
    def test_validate_sell_units_sufficient(self, trader, market):
        """Test _validate_sell_units returns True when units are sufficient."""
        market.step()
        trader.buy('AAPL', 10)
        
        result = trader._validate_sell_units('AAPL', 5)
        assert result is True
    
    def test_validate_sell_units_insufficient(self, trader, market):
        """Test _validate_sell_units returns False when units are insufficient."""
        market.step()
        trader.buy('AAPL', 10)
        
        result = trader._validate_sell_units('AAPL', 20)
        assert result is False
    
    def test_validate_sell_units_exact(self, trader, market):
        """Test _validate_sell_units with exact amount available."""
        market.step()
        trader.buy('AAPL', 10)
        
        result = trader._validate_sell_units('AAPL', 10)
        assert result is True
    
    def test_collect_positions_to_sell_full_position(self, trader, market):
        """Test _collect_positions_to_sell selling entire position."""
        market.step()
        trader.buy('AAPL', 10)
        
        initial_units = trader.portfolio_meta['AAPL']['units']
        positions = trader._collect_positions_to_sell('AAPL', 10)
        
        assert len(positions) == 1
        assert positions[0].units == 10
        assert trader.portfolio_meta['AAPL']['units'] == 0
        assert len(trader.portfolio['AAPL']) == 0
    
    def test_collect_positions_to_sell_partial_position(self, trader, market):
        """Test _collect_positions_to_sell selling part of a position."""
        market.step()
        trader.buy('AAPL', 10)
        
        positions = trader._collect_positions_to_sell('AAPL', 6)
        
        assert len(positions) == 1
        assert positions[0].units == 6
        assert trader.portfolio_meta['AAPL']['units'] == 4
        assert len(trader.portfolio['AAPL']) == 1
        assert trader.portfolio['AAPL'][0].units == 4
    
    def test_collect_positions_to_sell_multiple_positions(self, trader, market):
        """Test _collect_positions_to_sell across multiple positions."""
        market.step()
        trader.buy('AAPL', 10)
        market.step()
        trader.buy('AAPL', 8)
        
        positions = trader._collect_positions_to_sell('AAPL', 15)
        
        # Should sell entire first position (10) and part of second (5)
        assert len(positions) == 2
        assert positions[0].units == 10
        assert positions[1].units == 5
        assert trader.portfolio_meta['AAPL']['units'] == 3
    
    def test_process_sell_proceeds(self, trader):
        """Test _process_sell_proceeds updates liquid and fees correctly."""
        initial_liquid = trader.liquid
        initial_cumulative_fees = trader.cumulative_fees
        initial_cumulative_tax = trader.cumulative_tax
        
        trader._process_sell_proceeds(1500.0, 25.0, 50.0)
        
        # Liquid should increase by money - fee - tax
        assert trader.liquid == initial_liquid + 1500.0 - 25.0 - 50.0
        assert trader.liquid == initial_liquid + 1425.0
        assert trader.sell_fee == 25.0
        assert trader.tax == 50.0
        assert trader.cumulative_fees == initial_cumulative_fees + 25.0
        assert trader.cumulative_tax == initial_cumulative_tax + 50.0
    
    def test_log_sell_transaction(self, trader, market):
        """Test _log_sell_transaction creates correct transaction record."""
        market.step()
        price = market.get_stock_data('AAPL', 'Open')
        
        initial_count = len(trader.transaction_history)
        trader._log_sell_transaction('AAPL', 10, price, 1500.0, 25.0, 50.0)
        
        assert len(trader.transaction_history) == initial_count + 1
        transaction = trader.transaction_history[-1]
        assert transaction['type'] == 'SELL'
        assert transaction['ticker'] == 'AAPL'
        assert transaction['units'] == 10
        assert transaction['price'] == price
        assert transaction['total_value'] == 1500.0
        assert transaction['fee'] == 25.0
        assert transaction['tax'] == 50.0
    
    # Tests for update() micro-methods
    
    def test_calculate_portfolio_market_value_empty(self, trader):
        """Test _calculate_portfolio_market_value with empty portfolio."""
        market_value = trader._calculate_portfolio_market_value()
        assert market_value == 0.0
    
    def test_calculate_portfolio_market_value_single_ticker(self, trader, market):
        """Test _calculate_portfolio_market_value with one ticker."""
        market.step()
        trader.buy('AAPL', 10)
        
        market_value = trader._calculate_portfolio_market_value()
        expected_value = 10 * market.get_stock_data('AAPL', 'Open')
        assert market_value == expected_value
    
    def test_calculate_portfolio_market_value_multiple_tickers(self, trader, market):
        """Test _calculate_portfolio_market_value with multiple tickers."""
        market.step()
        trader.buy('AAPL', 10)
        trader.buy('GOOG', 5)
        
        market_value = trader._calculate_portfolio_market_value()
        expected_value = (10 * market.get_stock_data('AAPL', 'Open') +
                         5 * market.get_stock_data('GOOG', 'Open'))
        assert market_value == expected_value
    
    def test_calculate_portfolio_profit_no_fees(self, trader, market):
        """Test _calculate_portfolio_profit with no fees or tax."""
        market.step()
        trader.buy('AAPL', 10)
        trader.portfolio_market_value = trader._calculate_portfolio_market_value()
        
        profit = trader._calculate_portfolio_profit()
        
        # Profit should be market value - primary value - fees/tax
        expected_profit = (trader.portfolio_market_value -
                          trader.portfolio_primary_value -
                          trader.cumulative_fees -
                          trader.cumulative_tax)
        assert profit == expected_profit
    
    def test_calculate_portfolio_profit_with_fees(self, trader, market):
        """Test _calculate_portfolio_profit includes fees and tax."""
        market.step()
        trader.buy('AAPL', 10)
        
        # Simulate some fees and tax
        trader.cumulative_fees = 100.0
        trader.cumulative_tax = 50.0
        trader.portfolio_market_value = trader._calculate_portfolio_market_value()
        
        profit = trader._calculate_portfolio_profit()
        
        expected_profit = (trader.portfolio_market_value -
                          trader.portfolio_primary_value - 150.0)
        assert profit == expected_profit
    
    # Tests for step() micro-methods
    
    def test_reset_period_fees_and_tax(self, trader):
        """Test _reset_period_fees_and_tax resets and saves correctly."""
        trader.buy_fee = 25.0
        trader.sell_fee = 30.0
        trader.tax = 50.0
        
        initial_buy_history_len = len(trader.buy_fee_history)
        initial_sell_history_len = len(trader.sell_fee_history)
        initial_tax_history_len = len(trader.tax_history)
        
        buy_fee, sell_fee, tax = trader._reset_period_fees_and_tax()
        
        # Should return the values
        assert buy_fee == 25.0
        assert sell_fee == 30.0
        assert tax == 50.0
        
        # Should save to history
        assert len(trader.buy_fee_history) == initial_buy_history_len + 1
        assert len(trader.sell_fee_history) == initial_sell_history_len + 1
        assert len(trader.tax_history) == initial_tax_history_len + 1
        assert trader.buy_fee_history[-1] == 25.0
        assert trader.sell_fee_history[-1] == 30.0
        assert trader.tax_history[-1] == 50.0
        
        # Should reset to zero
        assert trader.buy_fee == 0
        assert trader.sell_fee == 0
        assert trader.tax == 0
    
    def test_calculate_yield_no_initial_value(self, trader):
        """Test _calculate_yield returns 0 when no initial value set."""
        yield_pct = trader._calculate_yield()
        assert yield_pct == 0.0
    
    def test_calculate_yield_with_profit(self, trader, market):
        """Test _calculate_yield calculates correct percentage."""
        market.step()
        trader.buy('AAPL', 10)
        trader.portfolio_initial_value = 1000.0
        trader.portfolio_market_value = 1200.0
        
        yield_pct = trader._calculate_yield()
        
        # (1200 / 1000 - 1) * 100 = 20%
        assert abs(yield_pct - 20.0) < 1e-10
    
    def test_calculate_yield_with_loss(self, trader, market):
        """Test _calculate_yield with negative return."""
        market.step()
        trader.buy('AAPL', 10)
        trader.portfolio_initial_value = 1000.0
        trader.portfolio_market_value = 800.0
        
        yield_pct = trader._calculate_yield()
        
        # (800 / 1000 - 1) * 100 = -20%
        assert abs(yield_pct - (-20.0)) < 1e-10
    
    def test_save_portfolio_history(self, trader, market):
        """Test _save_portfolio_history saves all required data."""
        market.step()
        test_date = date(2020, 1, 15)
        
        trader.liquid = 5000.0
        trader.portfolio_profit = 200.0
        trader.portfolio_market_value = 5200.0
        
        initial_liquid_len = len(trader.liquid_history)
        initial_profit_len = len(trader.profit_history)
        initial_value_len = len(trader.portfolio_value_history)
        initial_yield_len = len(trader.yield_history)
        initial_date_len = len(trader.date_history)
        
        trader._save_portfolio_history(test_date)
        
        # Check all histories were updated
        assert len(trader.liquid_history) == initial_liquid_len + 1
        assert len(trader.profit_history) == initial_profit_len + 1
        assert len(trader.portfolio_value_history) == initial_value_len + 1
        assert len(trader.yield_history) == initial_yield_len + 1
        assert len(trader.date_history) == initial_date_len + 1
        
        # Check correct values were saved
        assert trader.liquid_history[-1] == 5000.0
        assert trader.profit_history[-1] == 200.0
        assert trader.portfolio_value_history[-1] == 5200.0
        assert trader.date_history[-1] == test_date
        
        # Check initial value was set
        assert trader.portfolio_initial_value == 5200.0
    
    def test_save_portfolio_history_preserves_initial_value(self, trader, market):
        """Test _save_portfolio_history doesn't overwrite initial value."""
        market.step()
        test_date = date(2020, 1, 15)
        
        trader.portfolio_initial_value = 1000.0
        trader.portfolio_market_value = 1200.0
        
        trader._save_portfolio_history(test_date)
        
        # Initial value should not change
        assert trader.portfolio_initial_value == 1000.0


# Made with Bob
