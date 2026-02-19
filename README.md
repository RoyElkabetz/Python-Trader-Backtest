# Python Trader Backtest

A backtesting simulator for trading strategies using historical data from yfinance. Features portfolio rebalancing, multiple selling strategies (FIFO, LIFO, TAX_OPT), and comprehensive portfolio analytics.

---

## Installation

**Python 3.8+ required**

```bash
# Clone and navigate to project
git clone https://github.com/RoyElkabetz/Python-Trader-Backtest.git
cd Python-Trader-Backtest

# Create virtual environment
python3 -m venv .venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e ".[dev]"
```

---

## Quick Start

### Run Example
```bash
python examples/portfolio_comparison_demo.py
```

### Command Line
```bash
python src/main.py \
  -liquid 100000 \
  -tickers AAPL GOOG SPY \
  -periods 10 30 \
  -ratios 0.33 0.33 0.34 \
  -start_date 2023 1 1 \
  -end_date 2024 1 1
```

### Jupyter Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Python-Trader-Backtest/blob/main/notebooks/backtesting_simulator.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/RoyElkabetz/Python-Trader-Backtest/blob/main/notebooks/backtesting_simulator.ipynb)

---

## Features

### Selling Strategies
- **FIFO**: First In First Out
- **LIFO**: Last In First Out  
- **TAX_OPT**: Tax-optimized (minimizes capital gains)

### Portfolio Analytics
- **Sharpe Ratio**: Risk-adjusted returns
- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Portfolio standard deviation
- **Win Rate**: Percentage of profitable trades
- **Transaction History**: Complete audit trail

### Configurable Benchmark
Compare performance against S&P 500, NASDAQ, Dow Jones, or custom indices.

### Fee & Tax Modeling
- Percentage-based fees with minimum thresholds
- Capital gains tax on profits
- Accurate cost basis tracking

---

## Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `-liquid` | Initial capital | Yes | - |
| `-tickers` | Stock symbols (space-separated) | Yes | - |
| `-periods` | Rebalancing periods in days | Yes | - |
| `-ratios` | Portfolio weights (must sum to 1.0) | Yes | - |
| `-start_date` | Start date (YYYY M D) | Yes | - |
| `-end_date` | End date (YYYY M D) | Yes | - |
| `-deposit` | Periodic deposit amount | No | 0.0 |
| `-deposit_period` | Deposit frequency (days) | No | 30 |
| `-buy_fee` | Buy fee percentage | No | 0.08 |
| `-min_buy_fee` | Minimum buy fee | No | 2.0 |
| `-sell_fee` | Sell fee percentage | No | 0.08 |
| `-min_sell_fee` | Minimum sell fee | No | 2.0 |
| `-tax` | Capital gains tax percentage | No | 25.0 |
| `-sell_strategy` | FIFO, LIFO, or TAX_OPT | No | FIFO |
| `-verbose` | Print progress | No | True |

---

## Project Structure

```
Python-Trader-Backtest/
├── src/                    # Main package
│   ├── main.py            # CLI entry point
│   ├── markets.py         # Market data & benchmark
│   ├── brokers.py         # Fee/tax calculations
│   ├── traders.py         # Trading logic & analytics
│   ├── position.py        # Position tracking
│   ├── exceptions.py      # Custom exceptions
│   └── utils.py           # Plotting utilities
├── tests/                  # Pytest test suite (39 tests)
├── examples/               # Demo scripts
├── notebooks/              # Jupyter notebooks
└── pyproject.toml         # Package configuration
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## Recent Updates (2026)

### ✅ Major Refactoring
- **Position class**: 95%+ memory reduction, O(1) operations
- **Type hints**: Full type annotation coverage
- **Portfolio analytics**: Sharpe, CAGR, drawdown, volatility, win rate
- **Transaction history**: Complete audit trail with filtering
- **Configurable benchmark**: Compare against any index
- **Test suite**: 39 comprehensive pytest tests
- **Modern packaging**: Editable install with pyproject.toml

### ✅ Bug Fixes
- Fixed primary value tracking for multiple units
- Improved liquidity checks with fee accounting
- Resolved array/scalar conversion issues
- Updated for NumPy 1.21+ compatibility

---

## License

MIT License - See LICENSE file for details

---

**Happy Backtesting!** 📈
