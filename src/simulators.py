from .markets import Market
from .brokers import Broker
from .traders import Trader
from typing import List, Tuple
import copy as cp


def simulator(
    liquid: float,
    tickers: List[str],
    period: int,
    ratios: List[float],
    sell_strategy: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    buy_fee: float,
    min_buy_fee: float,
    sell_fee: float,
    min_sell_fee: float,
    tax: float,
    verbose: bool,
    deposit: float,
    deposit_period: int,
) -> Tuple[Trader, Market]:
    """
    Run a backtesting trading simulation with portfolio rebalancing.

    This function simulates trading strategies by creating a trader and balance it for a given period,
    executing trades, and tracking portfolio performance over time. It supports periodic deposits,
    automatic portfolio rebalancing, and various selling strategies.

    Args:
        liquid: Initial liquid cash available for trading.
        tickers: List of stock ticker symbols to trade.
        period: A single balance period (in days) to test for portfolio rebalancing.
        ratios: Target portfolio allocation ratios for each ticker (must sum to 1.0).
        sell_strategy: Strategy for selling stocks ('FIFO', 'LIFO', or 'TAX_OPT').
        start_date: Start date for simulation as [year, month, day].
        end_date: End date for simulation as [year, month, day].
        buy_fee: Buy transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_buy_fee: Minimum buy transaction fee in currency units.
        sell_fee: Sell transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_sell_fee: Minimum sell transaction fee in currency units.
        tax: Capital gains tax rate as a percentage (e.g., 25.0 for 25%).
        verbose: If True, print detailed trading information during simulation.
        deposit: Amount to deposit periodically (0 for no deposits).
        deposit_period: Number of days between periodic deposits.

    Returns:
        Returns a tuple of (trader, market) where:
            - trader: Is a Trader object.
            - market: The Market object used in the simulation.

    Example:
        >>> simulator(
        ...     liquid=10000.0,
        ...     tickers=['AAPL', 'GOOGL'],
        ...     period=30,
        ...     ratios=[0.5, 0.5],
        ...     sell_strategy='FIFO',
        ...     start_date=[2020, 1, 1],
        ...     end_date=[2021, 1, 1],
        ...     buy_fee=0.08,
        ...     min_buy_fee=2.0,
        ...     sell_fee=0.08,
        ...     min_sell_fee=2.0,
        ...     tax=25.0,
        ...     verbose=True,
        ...     deposit=1000.0,
        ...     deposit_period=30
        ... )
    """

    # init market and broker
    market = Market(tickers, start_date=start_date, end_date=end_date)
    broker = Broker(buy_fee=buy_fee, min_buy_fee=min_buy_fee, sell_fee=sell_fee,
                    min_sell_fee=min_sell_fee, tax=tax, my_market=market)

    # init new trader
    trader = Trader(liquid=liquid, balance_period=period, broker=broker, market=market,
                    verbose=verbose, sell_strategy=sell_strategy)

    # buy some stocks
    for ticker in tickers:
        trader.buy(ticker, 1)

    # update portfolio values after initial purchase
    trader.update()

    # balance portfolio with target ratios
    trader.balance(tickers, p=ratios)

    # set initial portfolio value for yield calculations
    if trader.portfolio_initial_value is None:
        trader.portfolio_initial_value = cp.copy(trader.portfolio_market_value)

    done = False
    steps = 0
    while not done:
        steps += 1
        if steps % 100 == 0:
            print('| Step: {:6.0f} / {:6.0f} | Balance period: {:4.0f} |'
                    .format(steps, market.steps, trader.balance_period))
        # step market forward in time
        done, previous_date = market.step()

        # step trader forward in time
        trader.step(previous_date)

        # deposit periodically
        if deposit > 0 and steps % deposit_period == 0:
            trader.deposit(deposit)
            trader.update()  # keep portfolio values current after deposit

        # balance trader portfolio
        if steps % trader.balance_period == 0:
            trader.balance(tickers, p=ratios)

    return trader, market

def multi_period_simulator(
    liquid: float,
    tickers: List[str],
    periods: List[int],
    ratios: List[float],
    sell_strategy: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    buy_fee: float,
    min_buy_fee: float,
    sell_fee: float,
    min_sell_fee: float,
    tax: float,
    verbose: bool,
    deposit: float,
    deposit_period: int,
) -> Tuple[List[Trader], Market]:
    """
    Run a backtesting trading simulation with portfolio rebalancing.

    This function simulates trading strategies by creating traders with different balance periods,
    executing trades, and tracking portfolio performance over time. It supports periodic deposits,
    automatic portfolio rebalancing, and various selling strategies.

    Args:
        liquid: Initial liquid cash available for trading.
        tickers: List of stock ticker symbols to trade.
        periods: List of balance periods (in days) to test for portfolio rebalancing.
        ratios: Target portfolio allocation ratios for each ticker (must sum to 1.0).
        sell_strategy: Strategy for selling stocks ('FIFO', 'LIFO', or 'TAX_OPT').
        start_date: Start date for simulation as [year, month, day].
        end_date: End date for simulation as [year, month, day].
        buy_fee: Buy transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_buy_fee: Minimum buy transaction fee in currency units.
        sell_fee: Sell transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_sell_fee: Minimum sell transaction fee in currency units.
        tax: Capital gains tax rate as a percentage (e.g., 25.0 for 25%).
        verbose: If True, print detailed trading information during simulation.
        deposit: Amount to deposit periodically (0 for no deposits).
        deposit_period: Number of days between periodic deposits.

    Returns:
        Returns a tuple of (trader, market) where:
            - trader: Is a Trader object.
            - market: The Market object used in the simulation.

    Example:
        >>> simulator(
        ...     liquid=10000.0,
        ...     tickers=['AAPL', 'GOOGL'],
        ...     periods=[30, 60],
        ...     ratios=[0.5, 0.5],
        ...     sell_strategy='FIFO',
        ...     start_date=[2020, 1, 1],
        ...     end_date=[2021, 1, 1],
        ...     buy_fee=0.08,
        ...     min_buy_fee=2.0,
        ...     sell_fee=0.08,
        ...     min_sell_fee=2.0,
        ...     tax=25.0,
        ...     verbose=True,
        ...     deposit=1000.0,
        ...     deposit_period=30
        ... )
    """
    traders_list = []

    for _, period in enumerate(periods):
        print(f'period: {period}')

        # run simulator for a single period value
        trader, market = simulator(
            liquid=liquid,
            tickers=tickers,
            period=period,
            ratios=ratios,
            sell_strategy=sell_strategy,
            start_date=start_date,
            end_date=end_date,
            buy_fee=buy_fee,
            min_buy_fee=min_buy_fee,
            sell_fee=sell_fee,
            min_sell_fee=min_sell_fee,
            tax=tax,
            verbose=verbose,
            deposit=deposit,
            deposit_period=deposit_period,
            )

        traders_list.append(trader)

    return traders_list, market
