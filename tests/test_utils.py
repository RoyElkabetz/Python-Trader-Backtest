"""
Tests for the utils.py plotting module.

Since these are plotting functions, tests focus on:
1. Functions execute without errors
2. Correct matplotlib objects are created
3. Functions handle edge cases properly
4. Functions work with different parameter combinations
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import date
from unittest.mock import patch, MagicMock
from src.utils import (
    market_plot,
    profit_and_portfolio_value,
    profits,
    portfolio_values,
    liquids,
    fees_and_tax,
    yields,
    yields_usd,
    plot_performance_metrics,
    print_performance_summary
)
from src.markets import Market
from src.brokers import Broker
from src.traders import Trader
from src.simulators import base_simulator


class TestMarketPlot:
    """Test suite for market_plot function."""
    
    @pytest.fixture
    def market(self):
        """Create a test market."""
        return Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 3, 31))
    
    @patch('matplotlib.pyplot.show')
    def test_market_plot_basic(self, mock_show, market):
        """Test basic market plot execution."""
        market_plot(market)
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_market_plot_with_specific_tickers(self, mock_show, market):
        """Test market plot with specific tickers."""
        market_plot(market, tickers=['AAPL'])
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_market_plot_without_normalization(self, mock_show, market):
        """Test market plot without normalization."""
        market_plot(market, normalize=False)
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_market_plot_different_parameter(self, mock_show, market):
        """Test market plot with different price parameter."""
        market_plot(market, prm='Close')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_market_plot_creates_figure(self, mock_show, market):
        """Test that market_plot creates a matplotlib figure."""
        market_plot(market)
        # Check that a figure was created
        assert len(plt.get_fignums()) > 0
        plt.close('all')


class TestTraderPlottingFunctions:
    """Test suite for trader-related plotting functions."""
    
    @pytest.fixture
    def traders_with_history(self):
        """Create traders with trading history for testing."""
        market = Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 6, 30))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        
        traders = []
        for period in [15, 30]:
            trader = Trader(
                liquid=10000,
                balance_period=period,
                ratios=[0.5, 0.5],
                deposit=0,
                deposit_period=30,
                broker=broker,
                market=market,
                verbose=False
            )
            
            # Run simulation to generate history
            market.reset()
            trader, _, _ = base_simulator(market, broker, trader, verbose=False)
            traders.append(trader)
        
        return traders
    
    @patch('matplotlib.pyplot.show')
    def test_profit_and_portfolio_value(self, mock_show, traders_with_history):
        """Test profit_and_portfolio_value plotting function."""
        profit_and_portfolio_value(
            traders_with_history,
            [15, 30],
            'Balance Period'
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_profits(self, mock_show, traders_with_history):
        """Test profits plotting function."""
        profits(traders_with_history, [15, 30], 'Balance Period')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_portfolio_values_with_colors(self, mock_show, traders_with_history):
        """Test portfolio_values with color palette."""
        portfolio_values(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_portfolio_values_without_colors(self, mock_show, traders_with_history):
        """Test portfolio_values without color palette."""
        portfolio_values(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=False
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_liquids(self, mock_show, traders_with_history):
        """Test liquids plotting function."""
        liquids(traders_with_history, [15, 30], 'Balance Period')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_fees_and_tax_with_colors(self, mock_show, traders_with_history):
        """Test fees_and_tax with color palette."""
        fees_and_tax(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_fees_and_tax_without_colors(self, mock_show, traders_with_history):
        """Test fees_and_tax without color palette."""
        fees_and_tax(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=False
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_with_colors(self, mock_show, traders_with_history):
        """Test yields plotting with color palette."""
        yields(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_without_colors(self, mock_show, traders_with_history):
        """Test yields plotting without color palette."""
        yields(
            traders_with_history,
            [15, 30],
            'Balance Period',
            use_colors=False
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_with_market_comparison(self, mock_show, traders_with_history):
        """Test yields plotting with market index comparison."""
        # Create market with index data
        market = Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 6, 30))
        
        yields(
            traders_with_history,
            [15, 30],
            'Balance Period',
            market=market,
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_usd_with_colors(self, mock_show, traders_with_history):
        """Test yields_usd plotting with color palette."""
        yields_usd(
            traders_with_history,
            [15, 30],
            'Balance Period',
            liquid=10000,
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_usd_without_colors(self, mock_show, traders_with_history):
        """Test yields_usd plotting without color palette."""
        yields_usd(
            traders_with_history,
            [15, 30],
            'Balance Period',
            liquid=10000,
            use_colors=False
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_usd_with_market_comparison(self, mock_show, traders_with_history):
        """Test yields_usd with market index comparison."""
        market = Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 6, 30))
        
        yields_usd(
            traders_with_history,
            [15, 30],
            'Balance Period',
            market=market,
            liquid=10000,
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')


class TestPerformanceMetrics:
    """Test suite for performance metrics plotting and printing."""
    
    @pytest.fixture
    def traders_with_metrics(self):
        """Create traders with complete metrics for testing."""
        market = Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 6, 30))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        
        traders = []
        for period in [15, 30, 60]:
            trader = Trader(
                liquid=10000,
                balance_period=period,
                ratios=[0.5, 0.5],
                deposit=0,
                deposit_period=30,
                broker=broker,
                market=market,
                verbose=False
            )
            
            # Run simulation
            market.reset()
            trader, _, _ = base_simulator(market, broker, trader, verbose=False)
            traders.append(trader)
        
        return traders
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_metrics_with_colors(self, mock_show, traders_with_metrics):
        """Test plot_performance_metrics with color palette."""
        plot_performance_metrics(
            traders_with_metrics,
            ['Period 15', 'Period 30', 'Period 60'],
            use_colors=True
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_metrics_without_colors(self, mock_show, traders_with_metrics):
        """Test plot_performance_metrics without color palette."""
        plot_performance_metrics(
            traders_with_metrics,
            ['Period 15', 'Period 30', 'Period 60'],
            use_colors=False
        )
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_metrics_creates_subplots(self, mock_show, traders_with_metrics):
        """Test that plot_performance_metrics creates correct subplot structure."""
        plot_performance_metrics(
            traders_with_metrics,
            ['Period 15', 'Period 30', 'Period 60'],
            use_colors=True
        )
        
        # Check that figure was created
        assert len(plt.get_fignums()) > 0
        plt.close('all')
    
    def test_print_performance_summary(self, traders_with_metrics, capsys):
        """Test print_performance_summary output."""
        print_performance_summary(
            traders_with_metrics,
            ['Period 15', 'Period 30', 'Period 60'],
            (2020, 1, 1),
            (2020, 6, 30),
            10000.0,
            30
        )
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Check that output contains expected elements
        assert 'PORTFOLIO PERFORMANCE SUMMARY' in captured.out
        assert 'Total Return' in captured.out
        assert 'CAGR' in captured.out
        assert 'Sharpe' in captured.out
        assert 'Volatility' in captured.out
        assert 'Max DD' in captured.out
        assert 'Period 15' in captured.out
        assert 'Period 30' in captured.out
        assert 'Period 60' in captured.out
        assert 'Simulation Period' in captured.out
        assert 'Initial Investment' in captured.out
        assert 'Rebalancing Period' in captured.out
    
    def test_print_performance_summary_formatting(self, traders_with_metrics, capsys):
        """Test that print_performance_summary formats numbers correctly."""
        print_performance_summary(
            traders_with_metrics,
            ['Test1', 'Test2', 'Test3'],
            (2020, 1, 1),
            (2020, 6, 30),
            10000.0,
            30
        )
        
        captured = capsys.readouterr()
        
        # Check for percentage signs
        assert '%' in captured.out
        # Check for dollar sign in initial investment
        assert '$' in captured.out
        # Check for separator lines
        assert '=' * 120 in captured.out
        assert '-' * 120 in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def single_trader(self):
        """Create a single trader for edge case testing."""
        market = Market(['AAPL'], (2020, 1, 1), (2020, 3, 31))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        trader = Trader(
            liquid=10000,
            balance_period=30,
            ratios=[1.0],
            deposit=0,
            deposit_period=30,
            broker=broker,
            market=market,
            verbose=False
        )
        
        trader, _, _ = base_simulator(market, broker, trader, verbose=False)
        return trader
    
    @patch('matplotlib.pyplot.show')
    def test_single_trader_plotting(self, mock_show, single_trader):
        """Test plotting functions with a single trader."""
        profits([single_trader], [30], 'Period')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_portfolio_values_single_trader(self, mock_show, single_trader):
        """Test portfolio_values with single trader."""
        portfolio_values([single_trader], [30], 'Period')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_metrics_single_trader(self, mock_show, single_trader):
        """Test plot_performance_metrics with single trader."""
        plot_performance_metrics([single_trader], ['Single'])
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_without_market(self, mock_show, single_trader):
        """Test yields plotting without market comparison."""
        yields([single_trader], [30], 'Period', market=None)
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_yields_usd_without_market(self, mock_show, single_trader):
        """Test yields_usd without market comparison."""
        yields_usd([single_trader], [30], 'Period', market=None)
        mock_show.assert_called_once()
        plt.close('all')
    
    def test_print_performance_summary_single_trader(self, single_trader, capsys):
        """Test print_performance_summary with single trader."""
        print_performance_summary(
            [single_trader],
            ['Single Strategy'],
            (2020, 1, 1),
            (2020, 3, 31),
            10000.0,
            30
        )
        
        captured = capsys.readouterr()
        assert 'Single Strategy' in captured.out
        assert 'PORTFOLIO PERFORMANCE SUMMARY' in captured.out


class TestPlottingIntegration:
    """Integration tests for plotting workflow."""
    
    @pytest.fixture
    def complete_simulation_traders(self):
        """Create traders from a complete simulation for integration testing."""
        market = Market(['AAPL', 'MSFT', 'GOOG'], (2020, 1, 1), (2020, 12, 31))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        
        traders = []
        for period in [20, 40]:
            trader = Trader(
                liquid=10000,
                balance_period=period,
                ratios=[0.33, 0.33, 0.34],
                deposit=500,
                deposit_period=30,
                broker=broker,
                market=market,
                verbose=False
            )
            
            market.reset()
            trader, _, _ = base_simulator(market, broker, trader, verbose=False)
            traders.append(trader)
        
        return traders, market
    
    @patch('matplotlib.pyplot.show')
    def test_complete_plotting_workflow(self, mock_show, complete_simulation_traders):
        """Test complete plotting workflow with all functions."""
        traders, market = complete_simulation_traders
        
        # Test all plotting functions in sequence
        profits(traders, [20, 40], 'Balance Period')
        portfolio_values(traders, [20, 40], 'Balance Period')
        liquids(traders, [20, 40], 'Balance Period')
        fees_and_tax(traders, [20, 40], 'Balance Period')
        yields(traders, [20, 40], 'Balance Period', market=market)
        yields_usd(traders, [20, 40], 'Balance Period', market=market)
        plot_performance_metrics(traders, ['Period 20', 'Period 40'])
        
        # Should have called show multiple times
        assert mock_show.call_count == 7
        plt.close('all')
    
    def test_complete_summary_workflow(self, complete_simulation_traders, capsys):
        """Test complete summary printing workflow."""
        traders, _ = complete_simulation_traders
        
        print_performance_summary(
            traders,
            ['Strategy A', 'Strategy B'],
            (2020, 1, 1),
            (2020, 12, 31),
            10000.0,
            30
        )
        
        captured = capsys.readouterr()
        
        # Verify comprehensive output
        assert 'Strategy A' in captured.out
        assert 'Strategy B' in captured.out
        assert all(metric in captured.out for metric in [
            'Total Return', 'CAGR', 'Sharpe', 'Volatility', 'Max DD'
        ])


class TestParameterVariations:
    """Test functions with various parameter combinations."""
    
    @pytest.fixture
    def traders_list(self):
        """Create a list of traders for parameter testing."""
        market = Market(['AAPL', 'MSFT'], (2020, 1, 1), (2020, 6, 30))
        broker = Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)
        
        traders = []
        for _ in range(3):
            trader = Trader(
                liquid=10000,
                balance_period=30,
                ratios=[0.5, 0.5],
                deposit=0,
                deposit_period=30,
                broker=broker,
                market=market,
                verbose=False
            )
            market.reset()
            trader, _, _ = base_simulator(market, broker, trader, verbose=False)
            traders.append(trader)
        
        return traders
    
    @patch('matplotlib.pyplot.show')
    def test_multiple_traders_plotting(self, mock_show, traders_list):
        """Test plotting with multiple traders."""
        portfolio_values(traders_list, [1, 2, 3], 'Test Parameter')
        mock_show.assert_called_once()
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_different_parameter_names(self, mock_show, traders_list):
        """Test plotting with different parameter names."""
        parameter_names = ['Alpha', 'Beta', 'Gamma']
        
        for param_name in parameter_names:
            profits(traders_list, [1, 2, 3], param_name)
        
        assert mock_show.call_count == len(parameter_names)
        plt.close('all')
    
    @patch('matplotlib.pyplot.show')
    def test_color_variations(self, mock_show, traders_list):
        """Test color parameter variations."""
        # Test with colors
        portfolio_values(traders_list, [1, 2, 3], 'Test', use_colors=True)
        
        # Test without colors
        portfolio_values(traders_list, [1, 2, 3], 'Test', use_colors=False)
        
        assert mock_show.call_count == 2
        plt.close('all')


# Made with Bob