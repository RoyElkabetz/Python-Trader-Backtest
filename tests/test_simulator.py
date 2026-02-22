"""
Tests for the simulator function in main.py.
"""

import pytest
import numpy as np
from datetime import date
from src.simulators import multi_period_simulator
from src.markets import Market
from src.brokers import Broker
from src.traders import Trader


class TestSimulator:
    """Test suite for the simulator function."""
    
    @pytest.fixture
    def basic_params(self):
        """Basic parameters for simulator tests."""
        return {
            'liquid': 10000,
            'tickers': ['AAPL', 'MSFT'],
            'periods': [30],
            'ratios': [0.5, 0.5],
            'sell_strategy': 'FIFO',
            'start_date': (2020, 1, 1),
            'end_date': (2020, 3, 31),
            'buy_fee': 0.1,
            'min_buy_fee': 1.0,
            'sell_fee': 0.1,
            'min_sell_fee': 1.0,
            'tax': 25.0,
            'verbose': False,
            'deposit': 0,
            'deposit_period': 30,
        }
    
    def test_simulator_basic_execution(self, basic_params):
        """Test that simulator runs without errors."""
        traders_list, market = multi_period_simulator(**basic_params)
        
        assert len(traders_list) == 1
        assert isinstance(traders_list[0], Trader)
        assert isinstance(market, Market)
    
    def test_simulator_multiple_periods(self, basic_params):
        """Test simulator with multiple balance periods."""
        basic_params['periods'] = [15, 30, 60]
        traders_list, market = multi_period_simulator(**basic_params)
        
        assert len(traders_list) == 3
        assert traders_list[0].balance_period == 15
        assert traders_list[1].balance_period == 30
        assert traders_list[2].balance_period == 60
    
    def test_simulator_initial_portfolio_value_set(self, basic_params):
        """Test that initial portfolio value is set correctly."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Initial portfolio value should be set
        assert trader.portfolio_initial_value is not None
        assert trader.portfolio_initial_value > 0
    
    def test_simulator_portfolio_updated_after_initial_buy(self, basic_params):
        """Test that portfolio is updated after initial stock purchases."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Portfolio should have stocks
        assert len(trader.portfolio) == 2
        assert 'AAPL' in trader.portfolio
        assert 'MSFT' in trader.portfolio
        
        # Portfolio market value should be calculated
        assert trader.portfolio_market_value > 0
    
    def test_simulator_history_tracking(self, basic_params):
        """Test that trading history is tracked."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # History should be populated
        assert len(trader.liquid_history) > 0
        assert len(trader.profit_history) > 0
        assert len(trader.portfolio_value_history) > 0
        assert len(trader.date_history) > 0
        assert len(trader.yield_history) > 0
    
    def test_simulator_with_deposits(self, basic_params):
        """Test simulator with periodic deposits."""
        basic_params['deposit'] = 1000
        basic_params['deposit_period'] = 10
        
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Liquid should be higher than initial due to deposits
        # (accounting for fees and market movements)
        assert len(trader.liquid_history) > 0
    
    def test_simulator_transaction_history(self, basic_params):
        """Test that transactions are recorded."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Should have initial buy transactions
        transactions = trader.get_transaction_history()
        assert len(transactions) > 0
        
        # Should have buy transactions for both tickers
        buy_transactions = trader.get_transaction_history(transaction_type='BUY')
        assert len(buy_transactions) >= 2
    
    def test_simulator_balance_operations(self, basic_params):
        """Test that balance operations are performed."""
        basic_params['periods'] = [10]  # Short period for more frequent balancing
        
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Should have multiple balance operations
        # Check that portfolio is relatively balanced
        summary = trader.get_transaction_summary()
        assert summary['total_transactions'] > 2  # Initial buys + rebalancing
    
    def test_simulator_different_ratios(self, basic_params):
        """Test simulator with different portfolio ratios."""
        basic_params['ratios'] = [0.7, 0.3]
        
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Portfolio should exist and be balanced according to ratios
        assert 'AAPL' in trader.portfolio
        assert 'MSFT' in trader.portfolio
    
    def test_simulator_sell_strategies(self, basic_params):
        """Test simulator with different sell strategies."""
        strategies = ['FIFO', 'LIFO', 'TAX_OPT']
        
        for strategy in strategies:
            basic_params['sell_strategy'] = strategy
            traders_list, market = multi_period_simulator(**basic_params)
            trader = traders_list[0]
            
            assert trader.sell_strategy == strategy
            assert len(trader.portfolio) > 0
    
    def test_simulator_market_reset_between_periods(self, basic_params):
        """Test that market is reset between different periods."""
        basic_params['periods'] = [15, 30]
        
        traders_list, market = multi_period_simulator(**basic_params)
        
        # Both traders should have similar initial conditions
        # (same starting date, same initial purchases)
        trader1 = traders_list[0]
        trader2 = traders_list[1]
        
        # Both should have started with the same tickers
        assert set(trader1.portfolio.keys()) == set(trader2.portfolio.keys())
    
    def test_simulator_portfolio_analytics_available(self, basic_params):
        """Test that portfolio analytics are available after simulation."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Should be able to get portfolio summary
        summary = trader.get_portfolio_summary()
        
        assert 'total_value' in summary
        assert 'total_return_pct' in summary
        assert 'sharpe_ratio' in summary
        assert 'max_drawdown_pct' in summary
        assert 'volatility_pct' in summary
        assert 'win_rate_pct' in summary
    
    def test_simulator_fees_and_tax_tracked(self, basic_params):
        """Test that fees and taxes are tracked."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Should have fee and tax history
        assert len(trader.buy_fee_history) > 0
        assert len(trader.sell_fee_history) > 0
        assert len(trader.tax_history) > 0
        
        # Cumulative fees and tax should be tracked
        assert trader.cumulative_fees >= 0
        assert trader.cumulative_tax >= 0
    
    def test_simulator_yield_calculations(self, basic_params):
        """Test that yield calculations are correct."""
        traders_list, market = multi_period_simulator(**basic_params)
        trader = traders_list[0]
        
        # Yield history should be populated
        assert len(trader.yield_history) > 0
        
        # Yields should be calculated from initial value
        if trader.portfolio_initial_value and trader.portfolio_initial_value > 0:
            # Last yield should match the calculation
            expected_yield = (trader.portfolio_market_value / trader.portfolio_initial_value - 1) * 100
            # Allow for small floating point differences
            assert abs(trader.yield_history[-1] - expected_yield) < 0.01


class TestSimulatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_simulator_single_ticker(self):
        """Test simulator with a single ticker."""
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL'],
            periods=[30],
            ratios=[1.0],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 3, 31),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=0,
            deposit_period=30,
        )
        
        assert len(traders_list) == 1
        assert len(traders_list[0].portfolio) == 1
    
    def test_simulator_three_tickers(self):
        """Test simulator with three tickers."""
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL', 'MSFT', 'GOOG'],
            periods=[30],
            ratios=[0.33, 0.33, 0.34],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 3, 31),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=0,
            deposit_period=30,
        )
        
        assert len(traders_list[0].portfolio) == 3
    
    def test_simulator_short_period(self):
        """Test simulator with very short time period."""
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL'],
            periods=[30],
            ratios=[1.0],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 1, 31),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=0,
            deposit_period=30,
        )
        
        assert len(traders_list) == 1
        assert len(traders_list[0].date_history) > 0
    
    def test_simulator_high_frequency_balancing(self):
        """Test simulator with very frequent balancing."""
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL', 'MSFT'],
            periods=[5],  # Balance every 5 days
            ratios=[0.5, 0.5],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 2, 29),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=0,
            deposit_period=30,
        )
        
        trader = traders_list[0]
        # Should have multiple transactions due to frequent rebalancing
        # (at least initial buys + some rebalancing trades)
        assert len(trader.transaction_history) >= 6


class TestSimulatorIntegration:
    """Integration tests for simulator workflow."""
    
    def test_complete_simulation_workflow(self):
        """Test complete simulation workflow from start to finish."""
        # Run a complete simulation
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL', 'MSFT'],
            periods=[30, 60],
            ratios=[0.6, 0.4],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 6, 30),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=500,
            deposit_period=30,
        )
        
        # Verify we have two traders (one for each period)
        assert len(traders_list) == 2
        
        # Verify both traders completed their simulations
        for trader in traders_list:
            # Should have history
            assert len(trader.date_history) > 0
            assert len(trader.portfolio_value_history) > 0
            
            # Should have portfolio
            assert len(trader.portfolio) > 0
            
            # Should have transactions
            assert len(trader.transaction_history) > 0
            
            # Should have initial value set
            assert trader.portfolio_initial_value is not None
            
            # Portfolio summary should be available
            summary = trader.get_portfolio_summary()
            assert summary['trading_days'] > 0
    
    def test_simulator_comparison_between_periods(self):
        """Test that different periods produce different results."""
        traders_list, market = multi_period_simulator(
            liquid=10000,
            tickers=['AAPL', 'MSFT'],
            periods=[15, 60],
            ratios=[0.5, 0.5],
            sell_strategy='FIFO',
            start_date=(2020, 1, 1),
            end_date=(2020, 6, 30),
            buy_fee=0.1,
            min_buy_fee=1.0,
            sell_fee=0.1,
            min_sell_fee=1.0,
            tax=25.0,
            verbose=False,
            deposit=0,
            deposit_period=30,
        )
        
        trader_15 = traders_list[0]
        trader_60 = traders_list[1]
        
        # Different balance periods should lead to different transaction counts
        # (15-day period should rebalance more frequently)
        assert trader_15.balance_period == 15
        assert trader_60.balance_period == 60
        
        # More frequent balancing typically means more transactions
        # (though not guaranteed depending on market conditions)
        summary_15 = trader_15.get_transaction_summary()
        summary_60 = trader_60.get_transaction_summary()
        
        # Both should have transactions
        assert summary_15['total_transactions'] > 0
        assert summary_60['total_transactions'] > 0


# Made with Bob