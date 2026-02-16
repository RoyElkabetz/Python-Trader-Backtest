# Python Trader Backtest

Python Trader Backtest is a backtesting simulator for simple trading strategies based on historical trading data from the yfinance Python package. The application enables variations of portfolio periodic balancing with weighted portfolio values for a variety of selling strategies: FIFO (First In First Out), LIFO (Last In First Out), or TAX_OPT that picks which stock to sell next by minimizing the amount of tax paid in the transaction.

---

## Requirements

Python 3.8 or higher is required.

### Installation

#### Using pip with virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Using pip directly:
```bash
pip install -r requirements.txt
```

### Dependencies

| Package    | Minimum Version |
|:----------:|:---------------:|
| pandas     | 1.3.0          |
| numpy      | 1.21.0         |
| matplotlib | 3.3.0          |
| yfinance   | 0.2.0          |

See `pyproject.toml` for complete package configuration.

---

## Notebook

| #   | file            | Subject                                         | Colab             | Nbviewer               |
|:----:|:--------------:|:------------------------------------------------:|:-----------------:|:---------------------:|
| 1   | `backtesting_simulator.ipynb` | Running the full simulator in Jupyter notebook   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Python-Trader-Backtest/blob/main/notebooks/backtesting_simulator.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/RoyElkabetz/Python-Trader-Backtest/blob/main/notebooks/backtesting_simulator.ipynb)|

---

## Command Line API

Run the `main.py` file in the `src` folder with the following arguments:

### Arguments

| Argument             | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `-tickers`            | The tickers to trade in, type=str, required=True, nargs='+'                     |
| `-periods`            | Periods to test balancing periods, type=int, required=True, nargs='+'                            |
| `-ratios`             | The balancing process happens according to the ratios, type=float, required=True, nargs='+'                             |
| `-start_date`         | Initial date of trading, type=int, required=True, nargs='+'                                                |
| `-end_date`           | Final date of trading, type=int, required=True, nargs='+'  |
| `-deposit`            | The amount to deposit in a periodic fashion, type=float, default=0.0                                    |
| `-deposit_period`     | The period of deposit, type=int, default=30                                                          |
| `-buy_fee`            | Transaction buying fee in percents, type=float, default=0.08                                              |
| `-min_buy_fee`        | Transaction minimal buying fee, type=float, default=2.                         |
| `-sell_fee`           | Transaction selling fee in percents, type=float, default=0.08|
| `-min_sell_fee`       | Transaction minimal selling fee, type=float, default=2. |
| `-tax`                | The amount of tax on profits in percents, type=float, default=25.                                         |
| `-liquid`             | The trader's initial liquid, type=float, required=True                                              |
| `-verbose`            | Print to terminal the balancing process (slows the simulation), type=bool, default=True                                              |
| `-plots_normalize`    | Normalizing the market plots to unity, type=bool, default=True                                                               |
| `-sell_strategy`      | The selling strategy for which the trader will follow, type=str, default='FIFO', choices='FIFO', 'LIFO', 'TAX_OPT'                   |

### Example Usage

```bash
# Navigate to the project directory
cd Python-Trader-Backtest

# Activate virtual environment (if using one)
source venv/bin/activate

# Run a simple backtest
python src/main.py \
  -liquid 100000 \
  -tickers AAPL GOOG SPY ORCL \
  -periods 2 10 20 \
  -ratios 0.25 0.25 0.25 0.25 \
  -deposit 1000 \
  -deposit_period 10 \
  -verbose False \
  -start_date 2023 1 1 \
  -end_date 2024 1 1
```

---

## Features

### Selling Strategies

- **FIFO (First In First Out)**: Sells the oldest stocks first
- **LIFO (Last In First Out)**: Sells the newest stocks first  
- **TAX_OPT (Tax Optimized)**: Sells stocks with the highest cost basis first to minimize capital gains tax

### Portfolio Balancing

The simulator supports periodic portfolio rebalancing according to specified ratios. This helps maintain target allocations across different stocks.

### Fee and Tax Modeling

- Configurable buy/sell fees (percentage-based with minimum thresholds)
- Capital gains tax on profits only (no tax on losses)
- Accurate tracking of cost basis for tax calculations

---

## Recent Updates (2026)

### ✅ Modernization
- Updated to work with latest Python (3.8+) and NumPy (1.21+)
- Fixed deprecated `np.int` and `np.float` usage
- Added `pyproject.toml` for modern Python packaging
- Improved compatibility with latest yfinance API

### ✅ Bug Fixes
- Fixed primary value tracking to correctly account for multiple units
- Improved buy liquidity check to account for transaction fees
- Fixed array/scalar conversion issues in formatting

### 🗑️ Removed Features
- Removed PySimpleGUI-based desktop app (command-line interface remains)

### 📊 Workflow Analysis
See `WORKFLOW_ANALYSIS.md` for detailed analysis of the trading workflow correctness, including fee calculation, tax computation, and profit tracking.

---

## Project Structure

```
Python-Trader-Backtest/
├── src/
│   ├── main.py          # Main entry point
│   ├── markets.py       # Market data handling
│   ├── brokers.py       # Fee and tax calculations
│   ├── traders.py       # Trading logic and portfolio management
│   └── utils.py         # Plotting and analysis utilities
├── notebooks/           # Jupyter notebooks for interactive use
├── requirements.txt     # Python dependencies
├── pyproject.toml      # Modern Python package configuration
└── README.md           # This file
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is open source and available under the MIT License.

---

Enjoy backtesting! 📈
