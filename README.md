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

## Logging

The system includes comprehensive logging to track all operations from market data fetching to individual trades and portfolio rebalancing.

### Quick Start

```bash
# Default logging (INFO to console, DEBUG to file)
python src/main.py --liquid 10000 --tickers AAPL GOOG --periods 30 --ratios 0.5 0.5 --start-date 2023 1 1 --end-date 2024 1 1

# Custom log levels
python src/main.py --log-level DEBUG --file-log-level INFO ...

# Disable console logging
python src/main.py --no-console-log ...

# Custom log file location
python src/main.py --log-dir ./my_logs --log-file my_backtest.log ...
```

### Logging Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--log-level` | Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--file-log-level` | File log level | DEBUG |
| `--log-file` | Log file name | trader_backtest.log |
| `--log-dir` | Log directory | logs/ |
| `--no-console-log` | Disable console logging | False |
| `--no-file-log` | Disable file logging | False |

### Log Output Examples

**Console (INFO level):**
```
2026-02-22 02:30:45 | INFO     | trader_backtest.markets | Initializing Market with 2 tickers: AAPL, GOOG
2026-02-22 02:30:46 | INFO     | trader_backtest.brokers | Broker initialized: buy_fee=0.08% (min $2.0), sell_fee=0.08% (min $2.0), tax=25%
2026-02-22 02:30:47 | INFO     | trader_backtest.traders | Trader initialized: liquid=$10000.00, balance_period=30, ratios=[0.5, 0.5]
2026-02-22 02:30:48 | INFO     | trader_backtest.simulators | Starting base simulator with 2 tickers: ['AAPL', 'GOOG']
```

**File (DEBUG level):**
```
2026-02-22 02:30:45.123 | DEBUG    | trader_backtest.markets | get_data_:67 | Fetching data for AAPL
2026-02-22 02:30:45.456 | DEBUG    | trader_backtest.brokers | buy_now:124 | Buy executed: AAPL x10 @ $150.00, total=$1500.00, fee=$2.00
2026-02-22 02:30:45.789 | DEBUG    | trader_backtest.traders | buy:244 | Attempting to buy AAPL: 10 units @ $150.00, estimated cost=$1500.00, fee=$2.00
```

### Log File Management

- **Rotating Files**: Logs automatically rotate when reaching 10MB
- **Backup Count**: Keeps 5 backup files (trader_backtest.log.1 through .5)
- **Location**: All logs stored in `logs/` directory by default

### Component Loggers

The system uses hierarchical loggers for each component:

- `trader_backtest.markets` - Market data fetching and date stepping
- `trader_backtest.brokers` - Trade execution, fees, and taxes
- `trader_backtest.traders` - Portfolio management and rebalancing
- `trader_backtest.simulators` - Simulation lifecycle and progress

### Configuration File (Optional)

Create `logging_config.yaml` for persistent settings:

```yaml
console_level: INFO
file_level: DEBUG
log_dir: logs
log_file: trader_backtest.log
max_bytes: 10485760  # 10 MB
backup_count: 5
```

---

## Command Line Arguments

### Trading Parameters

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--liquid` | Initial capital | Yes | - |
| `--tickers` | Stock symbols (space-separated) | Yes | - |
| `--periods` | Rebalancing periods in days | Yes | - |
| `--ratios` | Portfolio weights (must sum to 1.0) | Yes | - |
| `--start-date` | Start date (YYYY M D) | Yes | - |
| `--end-date` | End date (YYYY M D) | Yes | - |
| `--deposit` | Periodic deposit amount | No | 0.0 |
| `--deposit-period` | Deposit frequency (days) | No | 30 |
| `--buy-fee` | Buy fee percentage | No | 0.08 |
| `--min-buy-fee` | Minimum buy fee | No | 2.0 |
| `--sell-fee` | Sell fee percentage | No | 0.08 |
| `--min-sell-fee` | Minimum sell fee | No | 2.0 |
| `--tax` | Capital gains tax percentage | No | 25.0 |
| `--sell-strategy` | FIFO, LIFO, or TAX_OPT | No | FIFO |
| `--verbose` | Print progress | No | True |

### Logging Parameters

See [Logging](#logging) section above for logging-specific arguments.

---

## Project Structure

```
Python-Trader-Backtest/
├── src/                    # Main package
│   ├── main.py            # CLI entry point
│   ├── logging_config.py  # Centralized logging setup
│   ├── markets.py         # Market data & benchmark
│   ├── brokers.py         # Fee/tax calculations
│   ├── traders.py         # Trading logic & analytics
│   ├── simulators.py      # Simulation orchestration
│   ├── position.py        # Position tracking
│   ├── exceptions.py      # Custom exceptions
│   └── utils.py           # Plotting utilities
├── tests/                  # Pytest test suite (39 tests)
├── examples/               # Demo scripts
├── notebooks/              # Jupyter notebooks
├── logs/                   # Log files (auto-created)
├── logging_config.yaml     # Optional logging config
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

## License

MIT License - See LICENSE file for details

---
