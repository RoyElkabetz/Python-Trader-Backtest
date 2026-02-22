from .markets import Market
from .brokers import Broker
from .traders import Trader
from .logging_config import get_logger
from typing import List, Tuple
import copy as cp

logger = get_logger('simulators')


def base_simulator(
    market: Market,
    broker: Broker,
    trader: Trader,
    verbose: bool = True,
) -> Tuple[Trader, Broker, Market]:
    """
    Run a backtesting trading simulation with portfolio rebalancing.

    This function simulates trading strategies using pre-initialized market, broker, and trader objects.
    It executes trades, handles periodic deposits, performs automatic portfolio rebalancing, and tracks
    portfolio performance over time.

    Args:
        market: Market object containing ticker data and price history.
        broker: Broker object that handles transaction fees and tax calculations.
        trader: Trader object with initial configuration (liquid cash, balance period, ratios, etc.).
        verbose: If True, print detailed progress information during simulation. Default is True.

    Returns:
        Tuple[Trader, Broker, Market]: A tuple containing:
            - trader: The Trader object with updated portfolio and transaction history.
            - broker: The Broker object used in the simulation.
            - market: The Market object used in the simulation.

    Example:
        >>> from src.markets import Market
        >>> from src.brokers import Broker
        >>> from src.traders import Trader
        >>> market = Market(['AAPL', 'GOOGL'], start_date=[2020, 1, 1], end_date=[2021, 1, 1])
        >>> broker = Broker(buy_fee_percent=0.08, min_buy_fee=2.0, sell_fee_percent=0.08,
        ...                 min_sell_fee=2.0, tax=25.0, market=market)
        >>> trader = Trader(liquid=10000.0, balance_period=30, ratios=[0.5, 0.5],
        ...                 deposit=1000.0, deposit_period=30, broker=broker, market=market,
        ...                 verbose=True, sell_strategy='FIFO')
        >>> trader, broker, market = base_simulator(market, broker, trader, verbose=True)
    """

    logger.info(f"Starting base simulator with {len(market.tickers)} tickers: {market.tickers}")
    logger.info(f"Simulation period: {market.start_date} to {market.end_date} ({market.steps} trading days)")
    logger.info(f"Initial liquid: ${trader.liquid:.2f}, balance period: {trader.balance_period} days")
    
    # buy some stocks
    logger.debug("Purchasing initial positions (1 unit per ticker)")
    for ticker in market.tickers:
        trader.buy(ticker, 1)

    # update portfolio values after initial purchase
    trader.update()

    # balance portfolio with target ratios
    logger.info(f"Performing initial portfolio balance with ratios: {trader.ratios}")
    trader.balance(market.tickers, p=trader.ratios)

    # set initial portfolio value for yield calculations
    if trader.portfolio_initial_value is None:
        trader.portfolio_initial_value = cp.copy(trader.portfolio_market_value)
    
    logger.info(f"Initial portfolio value: ${trader.portfolio_initial_value:.2f}")

    done = False
    steps = 0
    while not done:
        steps += 1
        if steps % 100 == 0:
            progress_msg = f"Simulation progress: step {steps}/{market.steps}, balance period: {trader.balance_period}"
            if verbose:
                logger.info(progress_msg)
            else:
                logger.debug(progress_msg)
        
        # step market forward in time
        done, previous_date = market.step()

        # step trader forward in time
        trader.step(previous_date)

        # deposit periodically
        if trader.deposit > 0 and steps % trader.deposit_period == 0:
            logger.debug(f"Periodic deposit triggered at step {steps}")
            trader.make_deposit(trader.deposit)
            trader.update()  # keep portfolio values current after deposit

        # balance trader portfolio
        if steps % trader.balance_period == 0:
            logger.debug(f"Portfolio rebalancing triggered at step {steps}")
            trader.balance(market.tickers, p=trader.ratios)

    logger.info(f"Simulation complete after {steps} steps")
    logger.info(f"Final portfolio value: ${trader.portfolio_market_value:.2f}, liquid: ${trader.liquid:.2f}")
    
    return trader, broker, market

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
    Run multiple backtesting simulations with different balance periods.

    This function creates and runs multiple trading simulations, each with a different balance period,
    to compare portfolio performance across various rebalancing strategies. It initializes a single
    market and broker, then creates separate traders for each period, running complete simulations
    for each configuration.

    Args:
        liquid: Initial liquid cash available for trading.
        tickers: List of stock ticker symbols to trade.
        periods: List of balance periods (in days) to test for portfolio rebalancing.
        ratios: Target portfolio allocation ratios for each ticker (must sum to 1.0).
        sell_strategy: Strategy for selling stocks ('FIFO', 'LIFO', or 'TAX_OPT').
        start_date: Start date for simulation as tuple (year, month, day).
        end_date: End date for simulation as tuple (year, month, day).
        buy_fee: Buy transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_buy_fee: Minimum buy transaction fee in currency units.
        sell_fee: Sell transaction fee as a percentage (e.g., 0.08 for 0.08%).
        min_sell_fee: Minimum sell transaction fee in currency units.
        tax: Capital gains tax rate as a percentage (e.g., 25.0 for 25%).
        verbose: If True, print detailed trading information during simulation.
        deposit: Amount to deposit periodically (0 for no deposits).
        deposit_period: Number of days between periodic deposits.

    Returns:
        Tuple[List[Trader], Market]: A tuple containing:
            - traders_list: List of Trader objects, one for each balance period tested.
            - market: The Market object used in all simulations.

    Example:
        >>> traders, market = multi_period_simulator(
        ...     liquid=10000.0,
        ...     tickers=['AAPL', 'GOOGL'],
        ...     periods=[30, 60, 90],
        ...     ratios=[0.5, 0.5],
        ...     sell_strategy='FIFO',
        ...     start_date=(2020, 1, 1),
        ...     end_date=(2021, 1, 1),
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
    
    logger.info(f"Starting multi-period simulator with {len(periods)} periods: {periods}")
    logger.info(f"Tickers: {tickers}, ratios: {ratios}")
    logger.info(f"Initial liquid: ${liquid:.2f}, deposit: ${deposit:.2f} every {deposit_period} days")

    # init market and broker
    market = Market(tickers, start_date=start_date, end_date=end_date)
    broker = Broker(
        buy_fee_percent=buy_fee, min_buy_fee=min_buy_fee, sell_fee_percent=sell_fee,
        min_sell_fee=min_sell_fee, tax=tax, market=market
        )

    for idx, period in enumerate(periods):
        period_msg = f"Running simulation {idx+1}/{len(periods)} with balance period: {period} days"
        logger.info(period_msg)

        # reset market
        market.reset()

        # init new trader
        trader = Trader(
            liquid=liquid, balance_period=period, broker=broker, market=market,
            verbose=verbose, sell_strategy=sell_strategy, ratios=ratios,
            deposit=deposit, deposit_period=deposit_period,
            )

        # run simulator for a single period value
        trader, broker, market = base_simulator(
            market=market,
            broker=broker,
            trader=trader,
            verbose=verbose,
            )

        traders_list.append(trader)
        logger.info(f"Completed simulation {idx+1}/{len(periods)}, final value: ${trader.portfolio_market_value:.2f}")

    logger.info(f"Multi-period simulation complete. Tested {len(periods)} different balance periods")
    return traders_list, market
