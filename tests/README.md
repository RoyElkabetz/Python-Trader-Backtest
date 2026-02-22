# Test Suite for Python Trader Backtest

This directory contains the pytest-based test suite for the backtesting framework.

## Test Structure

- `test_position.py` - Tests for the Position dataclass
- `test_broker.py` - Tests for the Broker class
- `test_trader.py` - Tests for the Trader class, including:
  - Basic trading operations (buy/sell)
  - Portfolio analytics methods
  - Transaction history tracking
- `test_simulator.py` - Tests for the simulator function, including:
  - Basic simulation execution
  - Multiple period comparisons
  - Portfolio value tracking
  - Transaction history
  - Deposit functionality
  - Different sell strategies
  - Edge cases and integration tests
- `test_utils.py` - Tests for the plotting utilities module, including:
  - Market plotting functions
  - Trader performance visualization
  - Performance metrics plotting
  - Summary printing functions
  - Edge cases and parameter variations
- `test_logging.py` - Tests for the logging system

## Running Tests

### Run all tests:
```bash
pytest
```

### Run with verbose output:
```bash
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_position.py
```

### Run specific test class:
```bash
pytest tests/test_trader.py::TestPortfolioAnalytics
```

### Run specific test:
```bash
pytest tests/test_trader.py::TestTrader::test_buy_stock
```

### Run with coverage report:
```bash
pytest --cov=src --cov-report=html
```

## Test Coverage

The test suite covers:

✅ **Position Class**
- Position creation and properties
- Cost basis calculation
- Market value calculation
- Unrealized P&L calculation
- Price updates

✅ **Broker Class**
- Broker initialization and validation
- Buy operations and fee calculation
- Sell operations with tax calculation
- Minimum fee enforcement

✅ **Trader Class**
- Trading operations (buy/sell)
- Insufficient funds/shares handling
- Deposit and withdrawal
- Portfolio updates

✅ **Portfolio Analytics**
- Total return calculation
- Sharpe ratio
- Maximum drawdown
- CAGR (Compound Annual Growth Rate)
- Volatility
- Win rate
- Portfolio summary

✅ **Transaction History**
- Transaction recording
- Filtering by ticker and type
- Transaction summaries
- Per-ticker analysis

✅ **Simulator Function**
- Basic simulation execution
- Multiple balance periods
- Initial portfolio value setup
- Portfolio updates after purchases
- History tracking (liquid, profit, portfolio value, dates, yields)
- Periodic deposits with portfolio updates
- Transaction recording
- Balance operations
- Different portfolio ratios
- All sell strategies (FIFO, LIFO, TAX_OPT)
- Market reset between periods
- Portfolio analytics availability
- Fees and tax tracking
- Yield calculations
- Edge cases (single ticker, multiple tickers, short periods, high-frequency balancing)
- Integration tests (complete workflow, period comparisons)

✅ **Plotting Utilities (utils.py)**
- Market plotting (normalized and absolute values)
- Profit and portfolio value visualization
- Portfolio value history plotting
- Liquid history tracking
- Fees and tax visualization
- Yield plotting (with and without market comparison)
- Performance metrics comparison
- Summary printing and formatting
- Color palette variations
- Edge cases (single trader, multiple traders)
- Integration with complete simulation workflow

✅ **Logging System**
- Logger configuration and initialization
- Hierarchical logger creation
- File and console logging
- Different logging levels
- Integration across all components
- Complete workflow logging

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. Create test methods following the `test_*` naming convention
2. Use pytest fixtures for common setup
3. Use descriptive test names that explain what is being tested
4. Include docstrings explaining the test purpose
5. Use assertions to verify expected behavior

Example:
```python
def test_new_feature(self, trader):
    """Test description of what this tests."""
    # Arrange
    expected_value = 100
    
    # Act
    result = trader.new_method()
    
    # Assert
    assert result == expected_value
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src --cov-report=xml