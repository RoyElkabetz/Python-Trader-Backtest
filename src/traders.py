import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import date
from .markets import Market
from .brokers import Broker
from .exceptions import InsufficientFundsError, InsufficientSharesError
from .position import Position
import copy as cp

logger = logging.getLogger(__name__)


class Trader:
    """ A Trader class for Backtesting simulation of a periodic balancing strategy for stocks trading"""
    def __init__(self, liquid: float, balance_period: int, broker: Broker, market: Market,
                 verbose: bool = False, sell_strategy: str = 'FIFO') -> None:
        self.liquid = liquid
        self.balance_period = balance_period
        self.broker = broker
        self.market = market
        self.verbose = verbose

        assert sell_strategy in ['FIFO', 'LIFO', 'TAX_OPT'], \
            'sell_strategy should be one of the following: "FIFO", "LIFO", "TAX_OPT".'
        self.sell_strategy = sell_strategy

        # Trader's portfolio
        self.portfolio = {}
        self.portfolio_meta = {}
        self.portfolio_primary_value = 0
        self.portfolio_market_value = 0
        self.portfolio_profit = 0
        self.sell_fee = 0
        self.buy_fee = 0
        self.tax = 0
        self.fees_and_tax = 0
        self.cumulative_fees = 0.0  # Track cumulative fees incrementally
        self.cumulative_tax = 0.0   # Track cumulative tax incrementally
        self.usable_liquid = 0
        self.portfolio_initial_value = None

        # Save trading history
        self.liquid_history = []
        self.profit_history = []
        self.portfolio_value_history = []
        self.yield_history = []
        self.date_history = []
        self.error_history = []
        self.sell_fee_history = []
        self.buy_fee_history = []
        self.tax_history = []
        
        # Transaction history tracking
        self.transaction_history = []  # List of transaction dictionaries

    def buy(self, ticker: str, units: int) -> bool:
        """
        This function is used for buying new stocks and adding them to the trader's portfolio
        
        Args:
            ticker: The ticker of the stock
            units: Number of units to buy
            
        Returns:
            True if trade succeeded, False otherwise
        """
        ticker = ticker.upper()

        # get the stock current price
        price = self.market.get_stock_data(ticker, 'Open')

        # estimate total cost including fees
        total_cost = units * price
        estimated_fee = max(self.broker.buy_fee * total_cost, self.broker.min_buy_fee)
        
        # verify trader got enough liquid to complete the trade including fees
        if total_cost + estimated_fee > self.liquid:
            error_msg = f'Trader does not have enough liquid money to complete the {ticker} stock trade. Required: {total_cost + estimated_fee:.2f}, Available: {self.liquid:.2f}'
            logger.warning(error_msg)
            if self.verbose:
                print(f'\n[+][+] {error_msg}\n')
            return False
        else:
            # buy the stocks - now returns a Position object
            position, total_price, fee = self.broker.buy_now(ticker, units)
            self.buy_fee += fee
            self.cumulative_fees += fee  # Track cumulative fees

            # pay price
            self.liquid -= total_price

            # pay fee
            self.liquid -= fee

            # add ticker to portfolio
            if ticker not in self.portfolio:
                self.portfolio[ticker] = []
                self.portfolio_meta[ticker] = {'units': 0, 'sign': 0}

            # Add the Position object to portfolio
            self.portfolio[ticker].append(position)
            self.portfolio_meta[ticker]['units'] += units
            self.portfolio_primary_value += position.cost_basis
            
            # Log transaction
            self.transaction_history.append({
                'date': self.market.current_date,
                'type': 'BUY',
                'ticker': ticker,
                'units': units,
                'price': price,
                'total_value': total_price,
                'fee': fee,
                'tax': 0,
                'liquid_after': self.liquid
            })

            if self.verbose:
                total_price_val = total_price.item() if hasattr(total_price, 'item') else total_price
                fee_val = fee.item() if hasattr(fee, 'item') else fee
                print('[+] BUY  | Ticker: {:6s} | Units: {:4.0f} | Total price: {:10.2f} | Fee: {:8.2f} |'
                      .format(ticker, units, np.round(total_price_val, 2), np.round(fee_val, 2)))

            return True

    def sell(self, ticker: str, units: int) -> bool:
        """
        This function is used for selling stocks from the trader's portfolio
        
        Args:
            ticker: The ticker of the stock
            units: Number of units to sell
            
        Returns:
            True if trade succeeded, False otherwise
        """
        ticker = ticker.upper()

        # check trader got enough stocks to complete the sell
        if self.portfolio_meta[ticker]['units'] >= units:
            positions_to_sell = []
            units_remaining = units

            # remove positions from portfolio in a FIFO order (first in first out)
            while units_remaining > 0 and self.portfolio[ticker]:
                position = self.portfolio[ticker][0]  # Peek at first position
                
                if position.units <= units_remaining:
                    # Sell entire position
                    position = self.portfolio[ticker].pop(0)
                    units_remaining -= position.units
                    self.portfolio_meta[ticker]['units'] -= position.units
                    self.portfolio_primary_value -= position.cost_basis
                    positions_to_sell.append(position)
                else:
                    # Partial sale - split the position
                    units_to_sell = units_remaining
                    cost_basis_per_unit = position.purchase_price
                    
                    # Create a new position for the units being sold
                    sold_position = Position(
                        ticker=position.ticker,
                        units=units_to_sell,
                        purchase_price=position.purchase_price,
                        purchase_date=position.purchase_date,
                        current_price=position.current_price
                    )
                    positions_to_sell.append(sold_position)
                    
                    # Update the remaining position
                    position.units -= units_to_sell
                    self.portfolio_meta[ticker]['units'] -= units_to_sell
                    self.portfolio_primary_value -= cost_basis_per_unit * units_to_sell
                    units_remaining = 0

            # send positions to broker and collect money
            money, fee, tax = self.broker.sell_now(ticker, positions_to_sell)
            self.sell_fee += fee
            self.tax += tax
            self.cumulative_fees += fee  # Track cumulative fees
            self.cumulative_tax += tax   # Track cumulative tax

            # update the amount of liquid
            self.liquid += money - fee - tax
            
            # Log transaction
            price = self.market.get_stock_data(ticker, 'Open')
            self.transaction_history.append({
                'date': self.market.current_date,
                'type': 'SELL',
                'ticker': ticker,
                'units': units,
                'price': price,
                'total_value': money,
                'fee': fee,
                'tax': tax,
                'liquid_after': self.liquid
            })

            if self.verbose:
                money_val = money.item() if hasattr(money, 'item') else money
                fee_val = fee.item() if hasattr(fee, 'item') else fee
                tax_val = tax.item() if hasattr(tax, 'item') else tax
                print('[+] SELL | Ticker: {:6s} | Units: {:4.0f} | Total price: {:10.2f} | Fee: {:8.2f} '
                      '| Tax: {:8.2f} |'.format(ticker, units, np.round(money_val, 2), np.round(fee_val, 2), np.round(tax_val, 2)))

            return True
        else:
            error_msg = f'The trader does not have enough {ticker} units to complete the trade. Required: {units}, Available: {self.portfolio_meta[ticker]["units"]}'
            logger.warning(error_msg)
            if self.verbose:
                print(f'\n[+][+] {error_msg}\n')
            return False

    def update(self):
        """
        Function for updating the portfolio with the current market value of all stocks and computing the total profit
        :return: None
        """
        # update the portfolio market current prices
        self.portfolio_market_value = 0

        # update market prices for all owned stocks
        for ticker in self.portfolio:
            market_price = self.market.get_stock_data(ticker, 'Open')
            units = self.portfolio_meta[ticker]['units']
            self.portfolio_market_value += units * market_price

        # compute portfolio profit using cumulative tracking (O(1) instead of O(n))
        self.fees_and_tax = self.cumulative_fees + self.cumulative_tax
        self.portfolio_profit = self.portfolio_market_value - self.portfolio_primary_value - self.fees_and_tax

    def step(self, last_date):
        """
        Step one trading day ahead while updating the portfolio and saving portfolio history data for later analysis
        :param last_date: the current trading date
        :return: None
        """
        # update portfolio
        self.update()
        self.sort_tickers()

        # save trading history
        self.buy_fee_history.append(self.buy_fee)
        self.sell_fee_history.append(self.sell_fee)
        self.tax_history.append(self.tax)
        self.buy_fee = 0
        self.sell_fee = 0
        self.tax = 0
        self.liquid_history.append(self.liquid)
        self.profit_history.append(self.portfolio_profit)  # market value - value when bought - tax and fees
        self.portfolio_value_history.append(self.portfolio_market_value)
        if self.portfolio_initial_value is None:
            self.portfolio_initial_value = cp.copy(self.portfolio_market_value)
        self.yield_history.append((self.portfolio_market_value / self.portfolio_initial_value - 1.) * 100.)
        self.date_history.append(last_date)

    def _collect_portfolio_data(self, tickers):
        """
        Collect current portfolio data for all tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with portfolio data arrays
        """
        owned_units = np.zeros(len(tickers), dtype=int)
        market_value = np.zeros(len(tickers), dtype=float)
        owned_value = np.zeros(len(tickers), dtype=float)
        positions_buy_value = {}
        
        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_meta[ticker]['units']
            market_value[i] = self.market.get_stock_data(ticker, 'Open')
            owned_value[i] = owned_units[i] * market_value[i]
            positions_buy_value[ticker] = [pos.purchase_price for pos in self.portfolio[ticker]]
        
        return {
            'owned_units': owned_units,
            'market_value': market_value,
            'owned_value': owned_value,
            'positions_buy_value': positions_buy_value
        }
    
    def _calculate_tax_for_trades(self, tickers, units_to_trade, units_sign, market_value, positions_buy_value):
        """
        Calculate tax for a set of trades.
        
        Args:
            tickers: Array of ticker symbols
            units_to_trade: Array of units to trade (absolute values)
            units_sign: Array of trade direction (-1 for sell, 1 for buy)
            market_value: Array of current market prices
            positions_buy_value: Dict of purchase prices per ticker
            
        Returns:
            Array of tax amounts per ticker
        """
        tax = np.zeros(len(tickers), dtype=float)
        
        for i, ticker in enumerate(tickers):
            if units_sign[i] < 0:  # Selling
                total_market_value = units_to_trade[i] * market_value[i]
                total_owned_value = np.sum(positions_buy_value[ticker][:int(units_to_trade[i])])
                tax[i] = (total_market_value - total_owned_value) * self.broker.tax
        
        # Drop negative tax (from selling at a loss)
        return tax * (tax > 0)
    
    def _calculate_fees_for_trades(self, units_to_trade, units_sign, market_value):
        """
        Calculate fees for a set of trades.
        
        Args:
            units_to_trade: Array of units to trade (absolute values)
            units_sign: Array of trade direction (-1 for sell, 1 for buy)
            market_value: Array of current market prices
            
        Returns:
            Tuple of (sell_fee, buy_fee, total_fee)
        """
        sell_fee = np.max([
            np.sum(market_value * units_to_trade * (units_sign < 0)) * self.broker.sell_fee,
            np.sum(units_sign < 0) * self.broker.min_sell_fee
        ])
        buy_fee = np.max([
            np.sum(market_value * units_to_trade * (units_sign > 0)) * self.broker.buy_fee,
            np.sum(units_sign > 0) * self.broker.min_buy_fee
        ])
        return sell_fee, buy_fee, sell_fee + buy_fee
    
    def _calculate_target_units(self, owned_units, market_value, weights, usable_liquid):
        """
        Calculate target units for each ticker based on weights.
        
        Args:
            owned_units: Array of currently owned units
            market_value: Array of current market prices
            weights: Array of target weights
            usable_liquid: Total liquid available for rebalancing
            
        Returns:
            Tuple of (units_to_trade, trade_sign) where trade_sign is -1 for sell, 1 for buy
        """
        margins = market_value / 2
        value_to_target = usable_liquid * weights - margins
        units_target = np.array(np.round(value_to_target / market_value), dtype=int)
        units_to_trade = units_target - owned_units
        trade_sign = np.sign(units_to_trade)
        units_to_trade = np.abs(units_to_trade)
        
        return units_to_trade, trade_sign
    
    def _print_balance_info(self, tickers, owned_value, value_to_max, values_for_execution, market_value, execution_order):
        """Print verbose balance information."""
        liquid_val = self.liquid.item() if hasattr(self.liquid, 'item') else self.liquid
        print('[+] Liquid: {:14.2f} '.format(np.round(liquid_val, 2)))
        execute_str = ['[+] NEXT ']
        for ticker in tickers:
            execute_str.append('| ')
            execute_str.append(ticker)
            execute_str.append(': {:10.2f} ')
        execute_str.append('|')
        print('|-------------------------------------------------------------------------------------------------|')
        print(''.join(['[+] CURR '] + execute_str[1:]).format(*owned_value[execution_order]))
        print(''.join(['[+] GOAL '] + execute_str[1:]).format(*value_to_max[execution_order]))
        print(''.join(execute_str).format(*values_for_execution[execution_order]))
        print(''.join(['[+] UNIT '] + execute_str[1:]).format(*market_value[execution_order]))
        print('|-------------------------------------------------------------------------------------------------|')
    
    def balance(self, tickers: list, p=None):
        """
        Balance the trader's portfolio according to given weights.
        
        This method rebalances the portfolio by:
        1. Collecting current portfolio data
        2. Calculating target positions based on weights
        3. Estimating taxes and fees
        4. Executing trades in optimal order (sells before buys)
        5. Verifying the balance
        
        :param tickers: All the tickers in the portfolio (type: list)
        :param p: The weights for balancing with respect to the tickers order (type: list)
        :return: None
        """
        # Initialize weights and convert to numpy arrays
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=str)
        p = np.array(p, dtype=float)

        if self.verbose:
            print('\n')
            print('|------------------------------------------ BALANCING --------------------------------------------|')

        # Step 1: Collect current portfolio data
        portfolio_data = self._collect_portfolio_data(tickers)
        owned_units = portfolio_data['owned_units']
        market_value = portfolio_data['market_value']
        owned_value = portfolio_data['owned_value']
        positions_buy_value = portfolio_data['positions_buy_value']

        # Step 2: First iteration - estimate with mean balance
        margin = np.sum(market_value) / 2
        mean_balance = np.mean(owned_value) - margin
        units_to_mean = np.array(np.round((mean_balance - owned_value) / market_value), dtype=int)
        units_to_mean_sign = np.sign(units_to_mean)
        units_to_mean = np.abs(units_to_mean)

        tax = self._calculate_tax_for_trades(tickers, units_to_mean, units_to_mean_sign,
                                             market_value, positions_buy_value)
        _, _, total_fee = self._calculate_fees_for_trades(units_to_mean, units_to_mean_sign, market_value)

        # Calculate usable liquid after estimated costs
        self.usable_liquid = self.liquid + np.sum(owned_value) - np.sum(tax) - total_fee

        # Step 3: Calculate target positions based on weights
        units_to_max, units_to_max_sign = self._calculate_target_units(
            owned_units, market_value, p, self.usable_liquid
        )

        # Step 4: Refine with actual target calculations
        max_tax = self._calculate_tax_for_trades(tickers, units_to_max, units_to_max_sign,
                                                 market_value, positions_buy_value)
        _, _, max_total_fee = self._calculate_fees_for_trades(units_to_max, units_to_max_sign, market_value)

        # Recalculate usable liquid with refined estimates
        self.usable_liquid = self.liquid + np.sum(owned_value) - np.sum(max_tax) - max_total_fee

        # Recalculate target units with refined usable liquid
        units_to_max, units_to_max_sign = self._calculate_target_units(
            owned_units, market_value, p, self.usable_liquid
        )

        # Step 5: Sort operations (sells before buys)
        values_for_execution = units_to_max_sign * units_to_max * market_value
        execution_order = np.argsort(values_for_execution)
        tickers = tickers[execution_order]
        units_to_max_sign = units_to_max_sign[execution_order]
        units_to_max = units_to_max[execution_order]

        # Step 6: Print verbose information
        if self.verbose:
            margins = market_value / 2
            value_to_max = self.usable_liquid * p - margins
            self._print_balance_info(tickers, owned_value, value_to_max,
                                    values_for_execution, market_value, execution_order)

        # Step 7: Execute trades
        for i, ticker in enumerate(tickers):
            if units_to_max_sign[i] > 0:
                self.buy(ticker, units_to_max[i])
            elif units_to_max_sign[i] < 0:
                self.sell(ticker, units_to_max[i])

        # Step 8: Update and verify balance
        self.update()
        self.is_balanced(tickers, p=p[execution_order])

        if self.verbose:
            print('|-------------------------------------------------------------------------------------------------|')

    def is_balanced(self, tickers, p=None):
        """
        A function which checks if the trader's portfolio is in balance
        :param tickers: All the tickers in the portfolio (type: list)
        :param p: The weights for balancing with respect to the tickers order (type: list)
        :return: None
        """
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=str)
        p = np.array(p, dtype=float)

        # compute the owned value per ticker
        owned_units = np.zeros(len(tickers), dtype=int)
        market_value = np.zeros(len(tickers), dtype=float)

        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_meta[ticker]['units']
            market_value[i] = self.market.get_stock_data(ticker, 'Open')
        owned_value = owned_units * market_value

        # compute the half single unit margin error
        margins = market_value / 2
        allowed_margin = np.sum(margins)
        goal_values = self.usable_liquid * p - margins
        total_error = np.sum(np.abs(owned_value - goal_values))
        if self.verbose:
            print('| Current Error: {:10.2f} | Allowed Error: {:10.2f}'
                  .format(np.round(total_error), np.round(allowed_margin)))

        if total_error < allowed_margin:
            # the portfolio is balanced up to the allowed margin
            return True
        else:
            # the portfolio is not balanced
            return False

    def sort_tickers(self):
        """
        Sort positions in portfolio for each ticker in one of the following orders: FIFO, LIFO, TAX_OPT,
        where TAX_OPT will sort the positions according to their purchase prices, such that when sold would lead to
        a minimal tax payment.
        :return: None
        """
        # FIFO ordering of portfolio positions
        if self.sell_strategy == 'FIFO':
            return
        # LIFO ordering of portfolio positions (Last In First Out)
        elif self.sell_strategy == 'LIFO':
            for ticker in self.portfolio:
                positions = self.portfolio[ticker]
                # Sort by purchase date (newest first for LIFO)
                positions_with_dates = [(pos.purchase_date, pos) for pos in positions]
                positions_with_dates.sort(key=lambda x: x[0], reverse=True)
                self.portfolio[ticker] = [pos for _, pos in positions_with_dates]
        # TAX_OPT ordering of portfolio positions (sell highest cost basis first to minimize tax)
        elif self.sell_strategy == 'TAX_OPT':
            for ticker in self.portfolio:
                positions = self.portfolio[ticker]
                # Sort by purchase price (highest first to minimize capital gains)
                self.portfolio[ticker] = sorted(positions, key=lambda pos: pos.purchase_price, reverse=True)

    def deposit(self, amount):
        """
        Add money to the traders liquid
        :param amount: the amount of money to deposit
        :return: None
        """
        assert amount > 0, 'Trader can only deposit positive amounts of money.'
        self.liquid += amount

    def withdraw(self, amount: float) -> float:
        """
        Withdraw money from trader's liquid
        
        Args:
            amount: The amount of money to withdraw
            
        Returns:
            Amount withdrawn or 0 if insufficient funds
        """
        assert amount > 0, 'Trader can only withdraw positive amounts of money.'
        if self.liquid >= amount:
            self.liquid -= amount
            return amount
        else:
            error_msg = f'Trader does not have enough liquid (has {self.liquid:.2f} $) to withdraw {amount:.2f} $.'
            logger.warning(error_msg)
            if self.verbose:
                print(error_msg)
            return 0
    
    # ==================== Portfolio Analytics Methods ====================
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.
        
        The Sharpe ratio measures risk-adjusted returns by comparing excess returns
        to the standard deviation of returns.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # Calculate daily returns
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        # Calculate excess returns (assuming 252 trading days per year)
        daily_risk_free = risk_free_rate / 252
        excess_returns = returns - daily_risk_free
        
        # Calculate Sharpe ratio
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        # Annualize
        return sharpe * np.sqrt(252)
    
    def get_max_drawdown(self):
        """
        Calculate the maximum drawdown and its dates.
        
        Maximum drawdown is the largest peak-to-trough decline in portfolio value.
        
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if len(self.portfolio_value_history) < 2:
            return 0.0, None, None
        
        values = np.array(self.portfolio_value_history)
        
        # Calculate running maximum
        cummax = np.maximum.accumulate(values)
        
        # Calculate drawdown at each point
        drawdown = (values - cummax) / cummax
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find the peak before this drawdown
        peak_idx = np.argmax(values[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        return (
            max_dd * 100,  # As percentage
            self.date_history[peak_idx] if peak_idx < len(self.date_history) else None,
            self.date_history[max_dd_idx] if max_dd_idx < len(self.date_history) else None
        )
    
    def get_total_return(self) -> float:
        """
        Calculate total return percentage.
        
        Returns:
            Total return as a percentage
        """
        if self.portfolio_initial_value is None or self.portfolio_initial_value == 0:
            return 0.0
        
        current_total = self.portfolio_market_value + self.liquid
        initial_total = self.portfolio_initial_value + self.liquid_history[0] if self.liquid_history else self.portfolio_initial_value
        
        return ((current_total / initial_total) - 1) * 100
    
    def get_cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        CAGR represents the mean annual growth rate of the portfolio.
        
        Returns:
            CAGR as a percentage
        """
        if not self.date_history or len(self.date_history) < 2:
            return 0.0
        
        # Calculate time period in years
        days = (self.date_history[-1] - self.date_history[0]).days
        years = days / 365.25
        
        if years == 0:
            return 0.0
        
        # Calculate CAGR
        if self.portfolio_initial_value == 0:
            return 0.0
        
        current_total = self.portfolio_market_value + self.liquid
        initial_total = self.portfolio_initial_value + self.liquid_history[0] if self.liquid_history else self.portfolio_initial_value
        
        if initial_total == 0:
            return 0.0
        
        cagr = ((current_total / initial_total) ** (1 / years) - 1) * 100
        return cagr
    
    def get_volatility(self, annualized: bool = True) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            annualized: If True, return annualized volatility
            
        Returns:
            Volatility as a percentage
        """
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        volatility = np.std(returns) * 100
        
        if annualized:
            volatility *= np.sqrt(252)  # Annualize assuming 252 trading days
        
        return volatility
    
    def get_win_rate(self) -> float:
        """
        Calculate the percentage of profitable days.
        
        Returns:
            Win rate as a percentage
        """
        if len(self.profit_history) < 2:
            return 0.0
        
        # Calculate daily profit changes
        daily_profits = np.diff(self.profit_history)
        
        if len(daily_profits) == 0:
            return 0.0
        
        winning_days = np.sum(daily_profits > 0)
        total_days = len(daily_profits)
        
        return (winning_days / total_days) * 100
    
    def get_portfolio_summary(self) -> dict:
        """
        Get a comprehensive summary of portfolio performance.
        
        Returns:
            Dictionary with key performance metrics
        """
        max_dd, peak_date, trough_date = self.get_max_drawdown()
        
        return {
            # Current state
            'total_value': self.portfolio_market_value + self.liquid,
            'portfolio_value': self.portfolio_market_value,
            'liquid': self.liquid,
            'positions': sum(meta['units'] for meta in self.portfolio_meta.values()),
            
            # Returns
            'total_return_pct': self.get_total_return(),
            'cagr_pct': self.get_cagr(),
            
            # Risk metrics
            'sharpe_ratio': self.get_sharpe_ratio(),
            'volatility_pct': self.get_volatility(),
            'max_drawdown_pct': max_dd,
            'max_drawdown_peak_date': peak_date,
            'max_drawdown_trough_date': trough_date,
            
            # Trading metrics
            'win_rate_pct': self.get_win_rate(),
            'total_fees': sum(self.buy_fee_history) + sum(self.sell_fee_history),
            'total_tax': sum(self.tax_history),
            'total_costs': self.cumulative_fees + self.cumulative_tax,
            
            # Time period
            'start_date': self.date_history[0] if self.date_history else None,
            'end_date': self.date_history[-1] if self.date_history else None,
            'trading_days': len(self.date_history)
        }
    
    # ==================== Transaction History Methods ====================
    
    def get_transaction_history(self, ticker: Optional[str] = None,
                                transaction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get filtered transaction history.
        
        Args:
            ticker: Filter by ticker (optional)
            transaction_type: Filter by type 'BUY' or 'SELL' (optional)
            
        Returns:
            List of transaction dictionaries
        """
        transactions = self.transaction_history
        
        if ticker:
            transactions = [t for t in transactions if t['ticker'] == ticker.upper()]
        
        if transaction_type:
            transactions = [t for t in transactions if t['type'] == transaction_type.upper()]
        
        return transactions
    
    def get_transaction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transactions.
        
        Returns:
            Dictionary with transaction statistics
        """
        if not self.transaction_history:
            return {
                'total_transactions': 0,
                'total_buys': 0,
                'total_sells': 0,
                'total_buy_value': 0,
                'total_sell_value': 0,
                'total_fees_paid': 0,
                'total_tax_paid': 0,
                'unique_tickers': set()
            }
        
        buys = [t for t in self.transaction_history if t['type'] == 'BUY']
        sells = [t for t in self.transaction_history if t['type'] == 'SELL']
        
        return {
            'total_transactions': len(self.transaction_history),
            'total_buys': len(buys),
            'total_sells': len(sells),
            'total_buy_value': sum(t['total_value'] for t in buys),
            'total_sell_value': sum(t['total_value'] for t in sells),
            'total_fees_paid': sum(t['fee'] for t in self.transaction_history),
            'total_tax_paid': sum(t['tax'] for t in self.transaction_history),
            'unique_tickers': set(t['ticker'] for t in self.transaction_history),
            'first_transaction_date': self.transaction_history[0]['date'] if self.transaction_history else None,
            'last_transaction_date': self.transaction_history[-1]['date'] if self.transaction_history else None
        }
    
    def get_ticker_transactions(self, ticker: str) -> Dict[str, Any]:
        """
        Get all transactions for a specific ticker with analysis.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with ticker-specific transaction data
        """
        ticker = ticker.upper()
        transactions = self.get_transaction_history(ticker=ticker)
        
        if not transactions:
            return {
                'ticker': ticker,
                'total_transactions': 0,
                'buys': [],
                'sells': [],
                'total_units_bought': 0,
                'total_units_sold': 0,
                'net_units': 0,
                'total_invested': 0,
                'total_received': 0,
                'net_profit_loss': 0
            }
        
        buys = [t for t in transactions if t['type'] == 'BUY']
        sells = [t for t in transactions if t['type'] == 'SELL']
        
        total_units_bought = sum(t['units'] for t in buys)
        total_units_sold = sum(t['units'] for t in sells)
        total_invested = sum(t['total_value'] + t['fee'] for t in buys)
        total_received = sum(t['total_value'] - t['fee'] - t['tax'] for t in sells)
        
        return {
            'ticker': ticker,
            'total_transactions': len(transactions),
            'buys': buys,
            'sells': sells,
            'total_units_bought': total_units_bought,
            'total_units_sold': total_units_sold,
            'net_units': total_units_bought - total_units_sold,
            'total_invested': total_invested,
            'total_received': total_received,
            'net_profit_loss': total_received - total_invested,
            'average_buy_price': total_invested / total_units_bought if total_units_bought > 0 else 0,
            'average_sell_price': total_received / total_units_sold if total_units_sold > 0 else 0
        }
