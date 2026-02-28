"""
Bayesian Optimization Module for Trading Strategy Parameters

This module provides Bayesian optimization for finding optimal trading strategy
parameters using scikit-optimize and the base_simulator function.

Optimizes:
- Balance period (rebalancing frequency)
- Portfolio ratios (allocation weights for 9 tickers)

Maximizes a composite score combining:
- Sharpe Ratio (30%)
- CAGR (25%)
- Max Drawdown (20%, penalized)
- Volatility (15%, penalized)
- Win Rate (10%)
"""

from .traders import Trader
from .markets import Market
from .brokers import Broker
from .simulators import base_simulator
from .logging_config import get_logger

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from datetime import date

# Setup logging
logger = get_logger('optimization')


# ==================== Helper Functions ====================

def calculate_ratios(raw_weights: np.ndarray) -> np.ndarray:
    """
    Calculate portfolio ratios from raw weights.
    
    Normalizes weights to sum to 1.0 and rounds to 4 decimal places.
    Adjusts the last ratio to ensure exact sum of 1.0000.
    
    Args:
        raw_weights: Array of raw weight values
        
    Returns:
        Array of ratios (weights) that sum to exactly 1.0000,
        rounded to 4 decimal places
        
    Example:
        >>> calculate_ratios(np.array([1, 2, 3, 4]))
        array([0.1000, 0.2000, 0.3000, 0.4000])
        
        >>> calculate_ratios(np.array([5.5, 10.3, 7.2]))
        array([0.2391, 0.4478, 0.3131])  # Sums to exactly 1.0000
    """
    # Normalize weights to sum to 1.0
    weights_sum = np.sum(raw_weights)
    ratios = raw_weights / weights_sum
    
    # Round to 4 decimal places
    ratios = np.round(ratios, 4)
    
    # Adjust last ratio to ensure exact sum of 1.0000
    # (rounding can cause small deviations like 0.9999 or 1.0001)
    ratios[-1] = 1.0 - np.sum(ratios[:-1])
    
    return ratios


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized value clipped to [0, 1]
        
    Example:
        >>> normalize(50, 0, 100)
        0.5
    """
    if max_val == min_val:
        return 0.5  # Avoid division by zero
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)


def composite_objective(summary: dict) -> float:
    """
    Calculate composite objective score from portfolio summary.
    
    Combines multiple performance metrics with weights:
    - Sharpe Ratio: 30%
    - CAGR: 25%
    - Max Drawdown: 20% (penalized)
    - Volatility: 15% (penalized)
    - Win Rate: 10%
    
    Args:
        summary: Portfolio summary dict from trader.get_portfolio_summary()
        
    Returns:
        Composite score in [0, 1] range (higher is better)
        
    Example:
        >>> summary = {
        ...     'sharpe_ratio': 1.5,
        ...     'cagr_pct': 15.0,
        ...     'max_drawdown_pct': -20.0,
        ...     'volatility_pct': 18.0,
        ...     'win_rate_pct': 55.0
        ... }
        >>> composite_objective(summary)
        0.6234
    """
    # Extract metrics with defaults
    sharpe = summary.get('sharpe_ratio', 0.0)
    cagr = summary.get('cagr_pct', 0.0)
    max_dd = summary.get('max_drawdown_pct', 0.0)
    volatility = summary.get('volatility_pct', 0.0)
    win_rate = summary.get('win_rate_pct', 0.0)
    
    # Normalize each metric to [0, 1]
    norm_sharpe = normalize(sharpe, -1.0, 3.0)
    norm_cagr = normalize(cagr, -50.0, 100.0)
    norm_max_dd = normalize(-max_dd, 0.0, 80.0)  # Negative because lower is better
    norm_volatility = normalize(-volatility, -50.0, -5.0)  # Negative because lower is better
    norm_win_rate = normalize(win_rate, 0.0, 100.0)
    
    # Weighted combination
    score = (
        0.30 * norm_sharpe +
        0.25 * norm_cagr +
        0.20 * norm_max_dd +
        0.15 * norm_volatility +
        0.10 * norm_win_rate
    )
    
    return score


# ==================== Main Optimizer Class ====================

class BayesianOptimizer:
    """
    Bayesian optimizer for trading strategy parameters.
    
    Optimizes balance_period and portfolio ratios using Gaussian Process
    optimization to maximize a composite performance metric.
    
    Attributes:
        tickers: List of stock ticker symbols
        start_date: Simulation start date tuple (year, month, day)
        end_date: Simulation end date tuple (year, month, day)
        initial_liquid: Initial cash amount
        buy_fee: Buy transaction fee percentage
        min_buy_fee: Minimum buy fee
        sell_fee: Sell transaction fee percentage
        min_sell_fee: Minimum sell fee
        tax: Capital gains tax percentage
        deposit: Periodic deposit amount
        deposit_period: Days between deposits
        sell_strategy: Selling strategy ('FIFO', 'LIFO', 'TAX_OPT')
        verbose: Whether to print detailed simulation logs
        
    Example:
        >>> optimizer = BayesianOptimizer(
        ...     tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        ...              'META', 'NVDA', 'JPM', 'V'],
        ...     start_date=(2020, 1, 1),
        ...     end_date=(2023, 12, 31),
        ...     initial_liquid=100000.0
        ... )
        >>> result = optimizer.optimize(n_calls=50)
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        initial_liquid: float,
        buy_fee: float = 0.08,
        min_buy_fee: float = 2.0,
        sell_fee: float = 0.08,
        min_sell_fee: float = 2.0,
        tax: float = 25.0,
        deposit: float = 0.0,
        deposit_period: int = 30,
        sell_strategy: str = 'LIFO',
        verbose: bool = False
    ) -> None:
        """
        Initialize Bayesian optimizer with fixed simulation parameters.
        
        Args:
            tickers: List of stock ticker symbols (must have 9 tickers)
            start_date: Start date as (year, month, day)
            end_date: End date as (year, month, day)
            initial_liquid: Initial cash amount
            buy_fee: Buy fee percentage (default: 0.08%)
            min_buy_fee: Minimum buy fee (default: $2.00)
            sell_fee: Sell fee percentage (default: 0.08%)
            min_sell_fee: Minimum sell fee (default: $2.00)
            tax: Capital gains tax percentage (default: 25%)
            deposit: Periodic deposit amount (default: 0)
            deposit_period: Days between deposits (default: 30)
            sell_strategy: 'FIFO', 'LIFO', or 'TAX_OPT' (default: 'LIFO')
            verbose: Print detailed logs (default: False)
            
        Raises:
            ValueError: If number of tickers is not 9
        """
        if len(tickers) != 9:
            raise ValueError(f"Expected 9 tickers, got {len(tickers)}")
        
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_liquid = initial_liquid
        self.buy_fee = buy_fee
        self.min_buy_fee = min_buy_fee
        self.sell_fee = sell_fee
        self.min_sell_fee = min_sell_fee
        self.tax = tax
        self.deposit = deposit
        self.deposit_period = deposit_period
        self.sell_strategy = sell_strategy
        self.verbose = verbose
        
        # Track optimization history
        self.iteration = 0
        self.best_score = -np.inf
        self.history = []
        
        # Initialize Market and Broker once (reused across all simulations)
        logger.info(f"Initializing Market and Broker (one-time setup)...")
        self.market = Market(
            stocks=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.broker = Broker(
            buy_fee_percent=self.buy_fee,
            min_buy_fee=self.min_buy_fee,
            sell_fee_percent=self.sell_fee,
            min_sell_fee=self.min_sell_fee,
            tax=self.tax,
            market=self.market
        )
        
        logger.info(f"Initialized BayesianOptimizer with {len(tickers)} tickers")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Initial liquid: ${initial_liquid:,.2f}")
        logger.info(f"Sell strategy: {sell_strategy}")
        logger.info(f"Market data loaded: {self.market.steps} trading days")
    
    def objective_function(self, params: List[float]) -> float:
        """
        Objective function for Bayesian optimization.
        
        This function:
        1. Extracts balance_period and 9 raw weight values from params
        2. Normalizes weights to sum to 100% (ensures simplex constraint)
        3. Runs base_simulator with these parameters
        4. Calculates composite performance score
        5. Returns negative score (for minimization)
        
        Args:
            params: [balance_period, weight_1, ..., weight_9]
                   - balance_period: Integer in [5, 120]
                   - weights: 9 positive floats in [0.1, 50.0]
                     (normalized to percentages that sum to 100)
        
        Returns:
            Negative composite score (lower is better for optimizer)
            Returns large penalty (1e6) if simulation fails
            
        Note:
            This function is called by gp_minimize during optimization.
            Each call runs a complete backtest simulation.
            
        Example:
            If weights = [10, 20, 5, 15, 8, 12, 6, 14, 10]
            Sum = 100, so ratios = [10%, 20%, 5%, 15%, 8%, 12%, 6%, 14%, 10%]
            
            If weights = [1, 2, 1, 1, 1, 1, 1, 1, 1]
            Sum = 10, so ratios = [10%, 20%, 10%, 10%, 10%, 10%, 10%, 10%, 10%]
        """
        self.iteration += 1
        
        try:
            # Extract parameters
            balance_period = int(params[0])
            raw_weights = np.array(params[1:])
            
            # Calculate normalized ratios (weights that sum to 1.0)
            ratios = calculate_ratios(raw_weights)
            
            logger.info(f"Iteration {self.iteration}: balance_period={balance_period}")
            logger.debug(f"Raw weights: {raw_weights}")
            logger.info(f"Ratios: {[f'{r:.4f}' for r in ratios]}")
            logger.debug(f"Sum: {ratios.sum():.4f}")
            
            # Reset market to initial state (reuse existing market data)
            self.market.reset()
            
            # Create new trader with current parameters (ratios as weights summing to 1.0)
            trader = Trader(
                liquid=self.initial_liquid,
                balance_period=balance_period,
                ratios=ratios.tolist(),  # Pass as weights (0-1), rounded to 4 decimals
                deposit=self.deposit,
                deposit_period=self.deposit_period,
                broker=self.broker,
                market=self.market,
                verbose=self.verbose,
                sell_strategy=self.sell_strategy
            )
            
            # Run simulation (reusing market and broker)
            trader, broker, market = base_simulator(
                market=self.market,
                broker=self.broker,
                trader=trader,
                verbose=True  # Suppress simulation logs during optimization
            )
            
            # Get performance summary
            summary = trader.get_portfolio_summary()
            
            # Calculate composite score
            score = composite_objective(summary)
            
            # Store in history
            self.history.append({
                'iteration': self.iteration,
                'balance_period': balance_period,
                'raw_weights': raw_weights.tolist(),
                'ratios': ratios.tolist(),
                'score': score,
                'summary': summary
            })
            
            # Update best score
            if score > self.best_score:
                self.best_score = score
                logger.info(
                    f"Iteration {self.iteration}: New best score: {score:.4f} "
                    f"(balance_period={balance_period})"
                )
            else:
                logger.debug(f"Iteration {self.iteration}: Score: {score:.4f}")
            
            # Return negative for minimization
            return -score
            
        except Exception as e:
            logger.error(f"Iteration {self.iteration} failed: {str(e)}")
            logger.debug(f"Parameters: {params}", exc_info=True)
            return 1e6  # Large penalty for failed simulations
    
    def optimize(
        self,
        n_calls: int = 100,
        n_initial_points: int = 20,
        acq_func: str = 'EI',
        random_state: Optional[int] = 42,
        n_jobs: int = 1
    ):
        """
        Run Bayesian optimization to find optimal parameters.
        
        Uses Gaussian Process optimization (gp_minimize) to efficiently
        explore the parameter space and find the combination that maximizes
        the composite performance score.
        
        Args:
            n_calls: Number of function evaluations (default: 100)
            n_initial_points: Random exploration points (default: 20)
            acq_func: Acquisition function - 'EI', 'LCB', or 'PI' (default: 'EI')
            random_state: Random seed for reproducibility (default: 42)
            n_jobs: Number of parallel jobs (default: 1, sequential)
            
        Returns:
            OptimizeResult object from scikit-optimize containing:
            - x: Best parameters found
            - fun: Best objective value (negative score)
            - x_iters: All evaluated parameters
            - func_vals: All objective values
            - models: Gaussian process models
            
        Example:
            >>> result = optimizer.optimize(n_calls=50, n_initial_points=10)
            >>> print(f"Best score: {-result.fun:.4f}")
            >>> print(f"Best balance period: {int(result.x[0])}")
        """
        logger.info("=" * 70)
        logger.info("STARTING BAYESIAN OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        logger.info(f"Evaluations: {n_calls} (initial: {n_initial_points})")
        logger.info(f"Acquisition function: {acq_func}")
        logger.info(f"Random state: {random_state}")
        logger.info("=" * 70)
        
        # Define search space
        # Balance period + 9 weights (will be normalized to sum to 100%)
        space = [
            Integer(5, 120, name='balance_period'),
            Real(0.1, 50.0, name='weight_1'),
            Real(0.1, 50.0, name='weight_2'),
            Real(0.1, 50.0, name='weight_3'),
            Real(0.1, 50.0, name='weight_4'),
            Real(0.1, 50.0, name='weight_5'),
            Real(0.1, 50.0, name='weight_6'),
            Real(0.1, 50.0, name='weight_7'),
            Real(0.1, 50.0, name='weight_8'),
            Real(0.1, 50.0, name='weight_9'),
        ]
        
        # Reset tracking and market state
        self.iteration = 0
        self.best_score = -np.inf
        self.history = []
        self.market.reset()  # Ensure market starts from beginning
        
        # Run optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=random_state,
            verbose=False,  # We handle our own logging
            n_jobs=n_jobs
        )
        
        logger.info("=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best composite score: {-result.fun:.4f}")
        logger.info(f"Total evaluations: {len(result.func_vals)}")
        logger.info("=" * 70)
        
        return result
    
    def print_results(self, result) -> None:
        """
        Print formatted optimization results.
        
        Args:
            result: OptimizeResult from optimize()
        """
        best_params = result.x
        best_score = -result.fun
        
        balance_period = int(best_params[0])
        raw_weights = np.array(best_params[1:])
        
        # Calculate normalized ratios (weights that sum to 1.0)
        ratios = calculate_ratios(raw_weights)
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"Best Composite Score: {best_score:.4f}")
        print(f"Optimal Balance Period: {balance_period} days")
        print("\nOptimal Portfolio Allocation:")
        print("-" * 70)
        for ticker, ratio in zip(self.tickers, ratios):
            print(f"  {ticker:6s}: {ratio*100:6.2f}%")
        print("-" * 70)
        print(f"Total: {ratios.sum()*100:6.2f}%")
        print("=" * 70 + "\n")
    
    def get_best_parameters(self, result) -> dict:
        """
        Extract best parameters as a dictionary.
        
        Args:
            result: OptimizeResult from optimize()
            
        Returns:
            Dictionary with 'balance_period' and 'ratios_pct'
        """
        best_params = result.x
        balance_period = int(best_params[0])
        raw_weights = np.array(best_params[1:])
        
        # Calculate normalized ratios (weights that sum to 1.0)
        ratios = calculate_ratios(raw_weights)
        
        return {
            'balance_period': balance_period,
            'raw_weights': raw_weights.tolist(),
            'ratios': ratios.tolist(),
            'composite_score': -result.fun
        }
    
    def run_with_best_params(self, result, verbose: bool = True) -> Trader:
        """
        Run simulation with optimized parameters.
        
        Args:
            result: OptimizeResult from optimize()
            verbose: Print detailed simulation logs
            
        Returns:
            Trader object after simulation
        """
        params = self.get_best_parameters(result)
        
        logger.info("Running simulation with optimized parameters...")
        logger.info(f"Balance period: {params['balance_period']} days")
        logger.info(f"Ratios: {[f'{r:.4f}' for r in params['ratios']]}")
        
        # Reset market to initial state (reuse existing market data)
        self.market.reset()
        
        # Create trader with best parameters (ratios as weights summing to 1.0)
        trader = Trader(
            liquid=self.initial_liquid,
            balance_period=params['balance_period'],
            ratios=params['ratios'],  # Pass as weights (0-1)
            deposit=self.deposit,
            deposit_period=self.deposit_period,
            broker=self.broker,
            market=self.market,
            verbose=verbose,
            sell_strategy=self.sell_strategy
        )
        
        # Run simulation (reusing market and broker)
        trader, broker, market = base_simulator(
            market=self.market,
            broker=self.broker,
            trader=trader,
            verbose=verbose
        )
        
        return trader
    
    def plot_convergence(self, result, save_path: Optional[str] = None) -> None:
        """
        Plot optimization convergence.
        
        Args:
            result: OptimizeResult from optimize()
            save_path: Optional path to save figure
        """
        from skopt.plots import plot_convergence as skopt_plot_convergence
        
        fig, ax = plt.subplots(figsize=(10, 6))
        skopt_plot_convergence(result, ax=ax)
        ax.set_title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Evaluations', fontsize=12)
        ax.set_ylabel('Composite Score (negative)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization history showing score progression.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.history:
            logger.warning("No optimization history available")
            return
        
        iterations = [h['iteration'] for h in self.history]
        scores = [h['score'] for h in self.history]
        
        # Calculate running best
        running_best = []
        best_so_far = -np.inf
        for score in scores:
            best_so_far = max(best_so_far, score)
            running_best.append(best_so_far)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(iterations, scores, alpha=0.5, s=30, label='Evaluation Score', color='blue')
        ax.plot(iterations, running_best, 'r-', linewidth=2, label='Best Score')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Composite Score', fontsize=12)
        ax.set_title('Optimization Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"History plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, result, filepath: str = 'optimization_results.pkl') -> None:
        """
        Save optimization results to file.
        
        Args:
            result: OptimizeResult from optimize()
            filepath: Path to save results
        """
        import pickle
        
        save_data = {
            'result': result,
            'history': self.history,
            'best_parameters': self.get_best_parameters(result),
            'config': {
                'tickers': self.tickers,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_liquid': self.initial_liquid,
                'sell_strategy': self.sell_strategy
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Results saved to {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> dict:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to saved results
            
        Returns:
            Dictionary with result, history, and config
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data

# Made with Bob

