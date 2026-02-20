import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from .traders import Trader
from .markets import Market
import matplotlib.dates as mdates


def market_plot(market, prm='Open', tickers=None, normalize=True):
    data = market.stocks_data
    if tickers is None:
        tickers = list(data.keys())

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    ax.set_title('Market')
    for ticker in tickers:
        if normalize:
            ax.plot(data[ticker][prm] / data[ticker][prm].min(), label=ticker)
        else:
            ax.plot(data[ticker][prm], label=ticker)
    ax.legend()
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=70)
    if normalize:
        ax.set_ylabel('Normalized Value')
    else:
        ax.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.grid()
    plt.show()


def profit_and_portfolio_value(traders: list, parameter: list, parameter_name: str):
    interval = int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=150)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[0].set_title('profit history')

    for i, trader in enumerate(traders):
        axes[0].plot(trader.date_history, trader.profit_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[0].set_ylabel('USD')
    axes[0].legend()
    axes[0].grid()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[1].set_title('portfolio volume history')

    for i, trader in enumerate(traders):
        axes[1].plot(trader.date_history, trader.portfolio_value_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[1].set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes[1].legend()
    axes[1].grid()
    plt.show()


def profits(traders: list, parameter: list, parameter_name: str):
    interval = int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('profit history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.profit_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def portfolio_values(traders: list, parameter: list, parameter_name: str, use_colors=True):
    """
    Plot portfolio value history for multiple traders.
    
    Args:
        traders: List of Trader objects
        parameter: List of parameter values for labeling
        parameter_name: Name of the parameter being varied
        use_colors: If True, use color palette for better visualization
    """
    interval = int(len(traders[0].date_history) / 10)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.set_title('Portfolio Value History', fontsize=14, fontweight='bold', pad=20)
    
    # Use color palette if requested
    if use_colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traders)))
    else:
        colors = [None] * len(traders)
    
    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        ax.plot(trader.date_history, trader.portfolio_value_history,
                label=label, linewidth=2, color=colors[i], alpha=0.8)

    ax.set_ylabel('Portfolio Value (USD)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    fig.autofmt_xdate(bottom=0.15, rotation=30, ha='right')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def liquids(traders: list, parameter: list, parameter_name: str):
    interval = int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Liquid history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.liquid_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def fees_and_tax(traders: list, parameter: list, parameter_name: str, use_colors=True):
    """
    Plot cumulative fees and tax history for multiple traders.
    
    Args:
        traders: List of Trader objects
        parameter: List of parameter values for labeling
        parameter_name: Name of the parameter being varied
        use_colors: If True, use color palette for better visualization
    """
    interval = int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True, dpi=120)
    fig.suptitle('Trading Costs History', fontsize=14, fontweight='bold', y=0.995)
    
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[0].set_title('Cumulative Buy Fees', fontsize=12, pad=10)
    
    # Use color palette if requested
    if use_colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traders)))
    else:
        colors = [None] * len(traders)

    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        axes[0].plot(trader.date_history, np.cumsum(trader.buy_fee_history),
                     label=label, linewidth=2, color=colors[i], alpha=0.8)

    axes[0].set_ylabel('USD', fontsize=11, fontweight='bold')
    axes[0].legend(loc='best', framealpha=0.9, ncol=2, fontsize=7)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    axes[1].set_title('Cumulative Sell Fees', fontsize=12, pad=10)
    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        axes[1].plot(trader.date_history, np.cumsum(trader.sell_fee_history),
                     label=label, linewidth=2, color=colors[i], alpha=0.8)

    axes[1].set_ylabel('USD', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    axes[2].set_title('Cumulative Tax', fontsize=12, pad=10)
    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        axes[2].plot(trader.date_history, np.cumsum(trader.tax_history),
                     label=label, linewidth=2, color=colors[i], alpha=0.8)

    axes[2].set_ylabel('USD', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(bottom=0.08, rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


def yields(traders: list, parameter: list, parameter_name: str, market: Optional[Market] = None, use_colors=True):
    """
    Plot yield history for multiple traders.
    
    Args:
        traders: List of Trader objects
        parameter: List of parameter values for labeling
        parameter_name: Name of the parameter being varied
        market: Optional Market object for S&P 500 comparison
        use_colors: If True, use color palette for better visualization
    """
    interval = int(len(traders[0].date_history) / 10)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.set_title('Portfolio Yield History', fontsize=14, fontweight='bold', pad=20)
    
    # Use color palette if requested
    if use_colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traders)))
    else:
        colors = [None] * len(traders)

    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        ax.plot(trader.date_history, trader.yield_history,
                label=label, linewidth=2, color=colors[i], alpha=0.8)

    if market is not None and hasattr(market, 'index_data') and market.index_data is not None:
        ax.plot(market.index_data.index.to_numpy(), market.index_return_percent,
                label='S&P 500', linewidth=2, linestyle='--', color='black', alpha=0.6)
    
    ax.set_ylabel('Yield (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    fig.autofmt_xdate(bottom=0.15, rotation=30, ha='right')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def yields_usd(traders: list, parameter: list, parameter_name: str, market: Optional[Market] = None,
               liquid: float = 100000, use_colors=True):
    """
    Plot yield history in USD for multiple traders.
    
    Args:
        traders: List of Trader objects
        parameter: List of parameter values for labeling
        parameter_name: Name of the parameter being varied
        market: Optional Market object for S&P 500 comparison
        liquid: Initial liquid amount for S&P 500 comparison
        use_colors: If True, use color palette for better visualization
    """
    interval = int(len(traders[0].date_history) / 10)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.set_title('Portfolio Value History (USD)', fontsize=14, fontweight='bold', pad=20)
    
    # Use color palette if requested
    if use_colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traders)))
    else:
        colors = [None] * len(traders)

    for i, trader in enumerate(traders):
        label = f'{parameter_name}: {parameter[i]}'
        ax.plot(trader.date_history, trader.portfolio_value_history,
                label=label, linewidth=2, color=colors[i], alpha=0.8)

    if market is not None and hasattr(market, 'index_data') and market.index_data is not None:
        if hasattr(market, 'index_return_percent') and market.index_return_percent is not None:
            ax.plot(market.index_data.index.to_numpy(), (market.index_return_percent / 100 + 1) * liquid,
                    label='S&P 500', linewidth=2, linestyle='--', color='black', alpha=0.6)
    
    ax.set_ylabel('Portfolio Value (USD)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    fig.autofmt_xdate(bottom=0.15, rotation=30, ha='right')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_performance_metrics(traders: list, names: list, use_colors=True):
    """
    Plot comprehensive performance metrics comparison for multiple traders.
    
    Creates a 2x3 subplot grid showing:
    - Sharpe Ratio
    - Maximum Drawdown
    - CAGR (Compound Annual Growth Rate)
    - Annualized Volatility
    - Total Return
    - Risk-Return Scatter Plot
    
    Args:
        traders: List of Trader objects
        names: List of names/labels for each trader
        use_colors: If True, use color palette for better visualization
    """
    # Collect metrics for all traders
    sharpe_ratios = [t.get_sharpe_ratio() for t in traders]
    max_drawdowns = [t.get_max_drawdown()[0] for t in traders]
    cagrs = [t.get_cagr() for t in traders]
    volatilities = [t.get_volatility() for t in traders]
    total_returns = [t.get_total_return() for t in traders]
    
    # Use color palette if requested
    if use_colors:
        colors = plt.cm.tab10(np.linspace(0, 1, len(traders)))
    else:
        colors = ['steelblue'] * len(traders)
    
    # Create subplots for metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=120)
    fig.suptitle('Portfolio Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Sharpe Ratio
    axes[0, 0].bar(range(len(names)), sharpe_ratios, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Maximum Drawdown
    axes[0, 1].bar(range(len(names)), max_drawdowns, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Max Drawdown (%)', fontsize=10)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # CAGR
    axes[0, 2].bar(range(len(names)), cagrs, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[0, 2].set_title('Compound Annual Growth Rate (CAGR)', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('CAGR (%)', fontsize=10)
    axes[0, 2].set_xticks(range(len(names)))
    axes[0, 2].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Volatility
    axes[1, 0].bar(range(len(names)), volatilities, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('Annualized Volatility', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Volatility (%)', fontsize=10)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Total Return
    axes[1, 1].bar(range(len(names)), total_returns, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('Total Return', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Total Return (%)', fontsize=10)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Risk-Return Scatter
    axes[1, 2].scatter(volatilities, total_returns, c=colors, s=200, alpha=0.7,
                      edgecolors='black', linewidth=1.5)
    for i, name in enumerate(names):
        axes[1, 2].annotate(name, (volatilities[i], total_returns[i]),
                           fontsize=6, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')
    axes[1, 2].set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Volatility (%)', fontsize=10)
    axes[1, 2].set_ylabel('Total Return (%)', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def print_performance_summary(traders: list, names: list, start_date: tuple, end_date: tuple,
                             initial_investment: float, rebalancing_period: int):
    """
    Print a formatted table summarizing portfolio performance metrics.
    
    Args:
        traders: List of Trader objects
        names: List of names/labels for each trader
        start_date: Simulation start date tuple (year, month, day)
        end_date: Simulation end date tuple (year, month, day)
        initial_investment: Initial investment amount
        rebalancing_period: Rebalancing period in days
    """
    # Collect metrics
    sharpe_ratios = [t.get_sharpe_ratio() for t in traders]
    max_drawdowns = [t.get_max_drawdown()[0] for t in traders]
    cagrs = [t.get_cagr() for t in traders]
    volatilities = [t.get_volatility() for t in traders]
    total_returns = [t.get_total_return() for t in traders]
    
    print("\n" + "="*120)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*120)
    print(f"{'Portfolio':<25} {'Total Return':<15} {'CAGR':<12} {'Sharpe':<10} {'Volatility':<12} {'Max DD':<12}")
    print(f"{'Name':<25} {'(%)':<15} {'(%)':<12} {'Ratio':<10} {'(%)':<12} {'(%)':<12}")
    print("-"*120)
    
    for i, name in enumerate(names):
        print(f"{name:<25} {total_returns[i]:>13.2f}% {cagrs[i]:>10.2f}% {sharpe_ratios[i]:>9.2f} "
              f"{volatilities[i]:>10.2f}% {max_drawdowns[i]:>10.2f}%")
    
    print("="*120)
    print(f"\nSimulation Period: {start_date} to {end_date}")
    print(f"Initial Investment: ${initial_investment:,.2f}")
    print(f"Rebalancing Period: {rebalancing_period} days")
    print("="*120 + "\n")

