# Project Agentic Context

> **Purpose**: This document provides AI agents with structured context about the Python Trader Backtest system for efficient code understanding, modification, and debugging.

---

## 🎯 Quick Reference for AI Agents

### System Purpose
Stock trading backtesting simulator with portfolio rebalancing, realistic fees/taxes, and comprehensive analytics.

### Key Files to Read First
1. **`src/position.py`** - Core data structure (Position dataclass)
2. **`src/brokers.py`** - Transaction execution, fees, taxes
3. **`src/traders.py`** - Portfolio management logic
4. **`src/markets.py`** - Historical price data provider
5. **`src/simulators.py`** - Orchestration layer

### Critical Design Decision
**Position-Based Architecture** (NOT DataFrame-based):
- Each stock holding is a `Position` object (~200 bytes)
- 95%+ memory reduction vs DataFrame approach
- Type-safe with built-in cost basis tracking
- **NEVER** revert to DataFrame storage for positions

---

## 📊 Architecture Overview

### Component Hierarchy & Data Flow
```
┌─────────────────────────────────────────────────────────┐
│ Simulator (simulators.py)                              │
│ - Orchestrates entire backtest                         │
│ - Manages time stepping                                │
│ - Triggers rebalancing                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Trader (traders.py)                                     │
│ - Portfolio management                                  │
│ - Buy/sell decisions                                    │
│ - Performance metrics                                   │
│ - Transaction history                                   │
└────────────┬───────────────────────┬────────────────────┘
             │                       │
             ▼                       ▼
┌────────────────────────┐  ┌──────────────────────────┐
│ Broker (brokers.py)    │  │ Market (markets.py)      │
│ - Execute trades       │  │ - Historical prices      │
│ - Calculate fees       │  │ - Date management        │
│ - Calculate taxes      │  │ - Benchmark data         │
└────────────────────────┘  └──────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Position (position.py)                                  │
│ - Lightweight dataclass                                 │
│ - Cost basis calculation                                │
│ - Market value tracking                                 │
│ - P&L calculations                                      │
└─────────────────────────────────────────────────────────┘
```

### Separation of Concerns (CRITICAL)
Each component has a single responsibility:
- **Market**: ONLY provides price data and manages dates
- **Broker**: ONLY executes trades and calculates costs
- **Trader**: ONLY manages portfolio and strategy
- **Simulator**: ONLY orchestrates the simulation flow
- **Position**: ONLY stores individual position data

**⚠️ DO NOT mix responsibilities between components**

---

## 🏗️ Core Data Structures

### Position Object (position.py)
```python
self.portfolio = {
    'AAPL': [Position(...), Position(...), ...],  # List of Position objects
    'GOOGL': [Position(...), ...],
}

self.portfolio_meta = {
    'AAPL': {'units': 150, 'sign': 0},  # Metadata for quick lookups
    'GOOGL': {'units': 50, 'sign': 0},
}
```

#### Financial Tracking
```python
self.liquid                    # Available cash
self.portfolio_primary_value   # Cost basis (what was paid)
self.portfolio_market_value    # Current market value
self.portfolio_profit          # Total profit/loss
self.cumulative_fees           # All fees paid (buy + sell)
self.cumulative_tax            # All taxes paid
```

## Critical Implementation Rules

### 1. Fee Calculation
```python
# ALWAYS use this pattern for fees:
fee = fee_percent * transaction_value
if fee < min_fee:
    fee = min_fee
```
- Fees are percentage-based with minimum thresholds
- Both buy and sell operations have separate fee structures

### 2. Tax Calculation
```python
# ONLY tax profits, never losses:
tax = max(0, (sell_price - cost_basis) * tax_rate)
```
- Tax is calculated per transaction on capital gains only
- Use Position.cost_basis for accurate tracking

### 3. Buy Workflow (traders.py:223-270)

**Step-by-step process**:

1. **Calculate Cost** (traders.py:133-146):
   ```python
   total_cost = units * price
   estimated_fee = max(broker.buy_fee_percent * total_cost, broker.min_buy_fee)
   ```

2. **Validate Funds** (traders.py:148-164):
   ```python
   if total_cost + estimated_fee > self.liquid:
       return False  # Insufficient funds - MUST include fee
   ```

3. **Execute via Broker** (brokers.py:96-126):
   ```python
   position = Position(ticker, units, price, current_date, price)
   fee = max(buy_fee_percent * total_cost, min_buy_fee)
   return position, total_price, fee
   ```

4. **Update Portfolio** (traders.py:166-183):
   ```python
   portfolio[ticker].append(position)
   portfolio_meta[ticker]['units'] += units
   portfolio_primary_value += position.cost_basis  # CRITICAL
   ```

5. **Update Financials**:
   ```python
   liquid -= total_price + fee
   cumulative_fees += fee
   ```

**Common Mistakes to Avoid**:
- ❌ Forgetting to include fee in liquidity check
- ❌ Using `units * price` instead of `position.cost_basis`
- ❌ Not updating cumulative_fees

### 4. Sell Workflow (traders.py:394-432)

**Step-by-step process**:

1. **Validate Units** (traders.py:272-287):
   ```python
   if portfolio_meta[ticker]['units'] < units:
       return False  # Insufficient shares
   ```

2. **Collect Positions** (traders.py:289-335):
   ```python
   positions_to_sell = []
   units_remaining = units
   
   while units_remaining > 0:
       position = portfolio[ticker][0]  # First position (FIFO/LIFO/TAX_OPT sorted)
       
       if position.units <= units_remaining:
           # Sell entire position
           position = portfolio[ticker].pop(0)
           portfolio_primary_value -= position.cost_basis
           positions_to_sell.append(position)
           units_remaining -= position.units
       else:
           # Partial sale - split position
           sold_position = Position(ticker, units_remaining, position.purchase_price, ...)
           position.units -= units_remaining
           portfolio_primary_value -= sold_position.cost_basis
           positions_to_sell.append(sold_position)
           units_remaining = 0
   ```

3. **Execute via Broker** (brokers.py:128-158):
   ```python
   total_units = sum(pos.units for pos in positions_to_sell)
   cost_basis = sum(pos.cost_basis for pos in positions_to_sell)
   proceeds = current_price * total_units
   fee = max(sell_fee_percent * proceeds, min_sell_fee)
   tax = max(0, (proceeds - cost_basis) * tax_rate)
   return proceeds, fee, tax
   ```

4. **Update Financials** (traders.py:337-350):
   ```python
   liquid += proceeds - fee - tax
   cumulative_fees += fee
   cumulative_tax += tax
   ```

**Common Mistakes to Avoid**:
- ❌ Not handling partial position sales
- ❌ Forgetting to update portfolio_primary_value
- ❌ Not sorting positions before collection

### 5. Selling Strategies (traders.py:785-808)

**FIFO (First In First Out)** - Default:
```python
# Positions naturally ordered by purchase date (oldest first)
# No sorting needed - use natural order
```

**LIFO (Last In First Out)**:
```python
# Sort by purchase date, newest first
positions_with_dates = [(pos.purchase_date, pos) for pos in positions]
positions_with_dates.sort(key=lambda x: x[0], reverse=True)
portfolio[ticker] = [pos for _, pos in positions_with_dates]
```

**TAX_OPT (Tax Optimized)**:
```python
# Sort by purchase price, highest first (minimize capital gains)
portfolio[ticker] = sorted(positions, key=lambda pos: pos.purchase_price, reverse=True)
```

**When to Use Each**:
- **FIFO**: Default, simple, matches accounting standards
- **LIFO**: Minimize short-term capital gains in rising markets
- **TAX_OPT**: Minimize total tax liability (best for tax efficiency)

---

## 🧪 Testing & Validation

### Test Structure
- **Location**: `tests/` directory
- **Framework**: pytest
- **Coverage**: 39 tests covering all major components
- **Run**: `pytest` or `pytest --cov=src`

### Critical Test Areas
1. **Fee Calculations**: Percentage + minimum threshold logic
2. **Tax Calculations**: Only on profits, never on losses
3. **Portfolio Tracking**: Cost basis vs market value accuracy
4. **Selling Strategies**: FIFO, LIFO, TAX_OPT correctness
5. **Edge Cases**: Partial sales, insufficient funds, zero balances

### Test Fixtures (tests/test_trader.py)
```python
@pytest.fixture
def market():
    return Market(['AAPL', 'GOOG'], (2020, 1, 1), (2020, 6, 30))

@pytest.fixture
def broker(market):
    return Broker(0.1, 1.0, 0.1, 1.0, 25.0, market)

@pytest.fixture
def trader(broker, market):
    return Trader(liquid=10000, balance_period=30, ratios=[0.5, 0.5], ...)
```

---

## 🎨 Code Style & Conventions

### Method Organization
- **Public methods**: User-facing API (buy, sell, update, balance)
- **Private methods** (`_prefix`): Internal helpers, not for external use
- **Property methods** (`@property`): Computed values (cost_basis, market_value)

### Logging Pattern
```python
from .logging_config import get_logger
logger = get_logger('module_name')  # Use module name for hierarchy

# Log levels (use appropriately):
logger.debug("Detailed execution: var=%s", value)    # Development/debugging
logger.info("Important event: %s completed", action)  # Key state changes
logger.warning("Recoverable issue: %s", problem)      # User should know
logger.error("Serious problem: %s", error)            # System failure
```

### Error Handling Strategy
- **Custom Exceptions**: Use from `exceptions.py` (InvalidParameterError, InsufficientFundsError, etc.)
- **Validation**: Check parameters in `__init__` methods, raise exceptions early
- **Return Values**: Return `False` for failed operations (don't raise in normal trading flow)
- **Logging**: Log warnings for user-facing issues, errors for system problems

### Type Hints (REQUIRED)
```python
# ALWAYS include type hints for all functions:
def buy(self, ticker: str, units: int) -> bool:
    """Buy stock units."""
    ...

def _calculate_cost(self, units: int, price: float) -> Tuple[float, float]:
    """Calculate total cost and fee."""
    ...

# Use typing module for complex types:
from typing import List, Dict, Tuple, Optional
```

---

## 📋 Common Patterns & Idioms

### Accessing Market Data
```python
# Get current price for a ticker
price = self.market.get_stock_data(ticker, 'Open')

# Get all data for current date
data = self.market.get_stock_data(ticker, 'all')

# Current date
current_date = self.market.current_date
```

### Portfolio Iteration
```python
# Iterate over all positions
for ticker in self.portfolio:
    for position in self.portfolio[ticker]:
        # Process position
        cost = position.cost_basis
        value = position.market_value
```

### Safe Numeric Extraction
```python
# Handle pandas/numpy scalars
value = price.item() if hasattr(price, 'item') else price
```

---

## ⚠️ Common Pitfalls & How to Avoid Them

### ❌ DON'T Do These Things:

1. **Store DataFrames in portfolio**
   - Memory inefficient (10KB+ per position vs 200 bytes)
   - Use Position dataclass instead

2. **Forget fees in liquidity checks**
   ```python
   # WRONG:
   if units * price > self.liquid:
   
   # CORRECT:
   if units * price + estimated_fee > self.liquid:
   ```

3. **Calculate tax on losses**
   ```python
   # WRONG:
   tax = (sell_price - cost_basis) * tax_rate
   
   # CORRECT:
   tax = max(0, (sell_price - cost_basis) * tax_rate)
   ```

4. **Use price instead of cost_basis**
   ```python
   # WRONG:
   self.portfolio_primary_value += units * price
   
   # CORRECT:
   self.portfolio_primary_value += position.cost_basis
   ```

5. **Modify Position objects directly**
   ```python
   # WRONG:
   position.cost_basis = new_value
   
   # CORRECT:
   # cost_basis is a property, computed from units * purchase_price
   ```

6. **Mix component responsibilities**
   - Market should NOT execute trades
   - Broker should NOT manage portfolio
   - Trader should NOT fetch price data directly

### ✅ DO These Things:

1. **Use Position dataclass** for all stock holdings
2. **Validate funds** including estimated fees before buying
3. **Use Position.cost_basis** for accurate cost tracking
4. **Handle partial sales** by splitting Position objects
5. **Maintain cumulative tracking** for fees and taxes
6. **Keep separation of concerns** between components
7. **Read WORKFLOW_ANALYSIS.md** before making changes
8. **Run tests** after any modifications

---

## 🚀 Performance Considerations

### Memory Optimization
- **Position objects**: ~200 bytes each
- **DataFrames** (old approach): ~10KB+ each
- **Savings**: 95%+ memory reduction
- **Use portfolio_meta**: Quick unit lookups without iterating positions
- **Avoid copying**: Pass Position references, don't duplicate

### Computation Optimization
- **Cache market prices**: During update cycle, store in `current_data_cache`
- **Cumulative tracking**: Don't recalculate history, maintain running totals
- **Batch operations**: Process multiple tickers together when possible
- **Lazy evaluation**: Only calculate metrics when requested

---

## 🔧 Extension Points

### Adding New Selling Strategy
```python
# In traders.py, sort_tickers() method:
def sort_tickers(self):
    for ticker in self.portfolio:
        positions = self.portfolio[ticker]
        
        if self.sell_strategy == 'NEW_STRATEGY':
            # Implement your sorting logic
            self.portfolio[ticker] = sorted(positions, key=lambda pos: your_key)
```

### Adding New Performance Metric
```python
# In traders.py, follow this pattern:
def get_new_metric(self) -> float:
    """
    Calculate new performance metric.
    
    Returns:
        Metric value
    """
    # Use self.portfolio_value_history, self.date_history, etc.
    return calculated_value
```

### Adding New Order Type
```python
# In brokers.py, add new method:
def limit_buy(self, ticker: str, units: int, limit_price: float) -> Tuple[Position, float, float]:
    """Execute limit buy order."""
    current_price = self.market.get_stock_data(ticker, 'Open')
    if current_price <= limit_price:
        return self.buy_now(ticker, units)
    return None, 0, 0
```

---

## 📝 Documentation Standards

### Docstring Format (Google Style)
```python
def method_name(self, param1: str, param2: int) -> bool:
    """
    Brief one-line description.
    
    Longer description explaining the method's purpose, behavior,
    and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value and its meaning
        
    Raises:
        ExceptionType: When and why this exception is raised
        
    Example:
        >>> obj.method_name("AAPL", 10)
        True
    """
```

### Comment Guidelines
- **Explain WHY**, not WHAT (code should be self-documenting)
- **Complex calculations**: Add inline comments explaining the formula
- **Assumptions**: Document any assumptions or constraints
- **TODOs**: Use `# TODO:` for future improvements
- **References**: Link to docs or external resources when relevant

---

## 📁 Project File Structure

```
Python-Trader-Backtest/
├── src/                          # Main package
│   ├── __init__.py
│   ├── position.py              # ⭐ CORE: Position dataclass
│   ├── brokers.py               # Trade execution, fees, taxes
│   ├── traders.py               # Portfolio management, strategies
│   ├── markets.py               # Historical price data, benchmarks
│   ├── simulators.py            # Backtesting orchestration
│   ├── exceptions.py            # Custom exception classes
│   ├── logging_config.py        # Centralized logging setup
│   ├── utils.py                 # Plotting and utility functions
│   └── main.py                  # CLI entry point
│
├── tests/                        # Test suite (39 tests)
│   ├── test_position.py         # Position dataclass tests
│   ├── test_broker.py           # Broker transaction tests
│   ├── test_trader.py           # Trader portfolio tests
│   ├── test_simulator.py        # Simulation tests
│   └── ...
│
├── docs/                         # Documentation
│   ├── PROJECT_AGENTIC_CONTEXT.md  # ⭐ This file (AI agent guide)
│   └── WORKFLOW_ANALYSIS.md        # Correctness analysis
│
├── examples/                     # Demo scripts
│   └── portfolio_comparison_demo.py
│
├── notebooks/                    # Jupyter notebooks
│   └── backtesting_simulator.ipynb
│
├── logs/                         # Log files (auto-created)
├── pyproject.toml               # Package configuration
├── pytest.ini                   # Test configuration
├── logging_config.yaml          # Optional logging config
└── README.md                    # User documentation
```

---

## 📦 Dependencies

### Core Dependencies
- **pandas** (>=1.3.0): Market data handling, time series
- **numpy** (>=1.21.0): Numerical calculations, array operations
- **yfinance** (>=0.1.70): Historical stock data download from Yahoo Finance

### Development Dependencies
- **pytest** (>=7.0.0): Testing framework
- **pytest-cov**: Code coverage reporting
- **PyYAML**: Configuration file parsing

### Installation
```bash
pip install -e ".[dev]"  # Install with dev dependencies
```

---

## 🔑 Quick Reference: Key Formulas

### Portfolio Profit
```python
profit = portfolio_market_value - portfolio_primary_value - cumulative_fees - cumulative_tax
```
Where:
- `portfolio_market_value`: Current value of all positions
- `portfolio_primary_value`: Total cost basis (what was paid)
- `cumulative_fees`: All buy + sell fees
- `cumulative_tax`: All capital gains taxes

### Transaction Fee
```python
fee = max(fee_percent * transaction_value, min_fee)
```
Example: 0.08% fee with $2 minimum on $1000 trade = max($0.80, $2) = $2

### Capital Gains Tax
```python
tax = max(0, (sell_price - cost_basis) * tax_rate)
```
Only applied on profits, never on losses

### Cost Basis
```python
cost_basis = units * purchase_price  # Use Position.cost_basis property
```

---

## 🤖 AI Agent Workflow Guide

### 🔍 When Analyzing Code:
1. **Start with** `src/position.py` - understand the core data model
2. **Then read** `src/brokers.py` - see how transactions work
3. **Next review** `src/traders.py` - understand portfolio logic
4. **Check** `src/markets.py` - see data provider interface
5. **Finally** `src/simulators.py` - understand orchestration

### ✏️ When Making Changes:
1. **Read** `docs/WORKFLOW_ANALYSIS.md` first (understand current correctness)
2. **Maintain** Position-based architecture (never revert to DataFrames)
3. **Preserve** separation of concerns (don't mix component responsibilities)
4. **Keep** cumulative tracking intact (fees and taxes)
5. **Follow** existing patterns and conventions
6. **Add** tests for new functionality
7. **Update** documentation if API changes
8. **Run** `pytest` to verify correctness

### 🐛 When Debugging:
1. **Check** Position.cost_basis calculations first
2. **Verify** fee inclusion in liquidity checks
3. **Confirm** tax only applies to profits (max(0, ...))
4. **Review** selling strategy implementation (FIFO/LIFO/TAX_OPT)
5. **Examine** cumulative tracking accuracy
6. **Inspect** portfolio_primary_value updates
7. **Validate** partial position sale handling

### 🎯 When Optimizing:
1. **Profile** before optimizing (don't guess)
2. **Use** portfolio_meta for quick lookups
3. **Cache** market data during update cycles
4. **Avoid** unnecessary Position object copies
5. **Batch** operations when possible
6. **Maintain** memory efficiency (Position-based design)

---

## 📊 Performance Metrics Available

The Trader class provides these analytics methods:
- `get_sharpe_ratio()`: Risk-adjusted returns
- `get_max_drawdown()`: Largest peak-to-trough decline
- `get_cagr()`: Compound Annual Growth Rate
- `get_volatility()`: Annualized portfolio volatility
- `get_win_rate()`: Percentage of profitable trades
- `get_total_return()`: Total return percentage
- `get_portfolio_summary()`: Comprehensive summary dict
- `get_transaction_history()`: All buy/sell transactions
- `get_ticker_transactions(ticker)`: Transactions for specific ticker

---

## 🏷️ Version & Status

**Current Version**: 2.0 (Position-based architecture)
**Status**: ✅ Production-Ready
**Last Updated**: 2026-02-22
**Test Coverage**: 39 tests passing
**Known Issues**: None

### Version History
- **v1.0** (Deprecated): DataFrame-based storage
  - Memory inefficient (~10KB per position)
  - Workflow issues with cost basis tracking
  
- **v2.0** (Current): Position-based architecture
  - 95%+ memory reduction (~200 bytes per position)
  - All workflow issues fixed
  - Type-safe implementation
  - Production-ready

---

## 📚 Additional Resources

- **README.md**: User-facing documentation, installation, CLI usage
- **WORKFLOW_ANALYSIS.md**: Detailed correctness analysis of all workflows
- **examples/**: Working code examples
- **notebooks/**: Interactive Jupyter notebooks
- **tests/**: Comprehensive test suite with examples

---

**For AI Agents**: This document is optimized for quick understanding and safe code modifications. Always prioritize correctness over cleverness, and maintain the Position-based architecture at all costs.