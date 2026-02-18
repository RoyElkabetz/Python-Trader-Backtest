# Examples Directory

This directory contains example scripts demonstrating how to use the Python Trader Backtest framework.

## Available Examples

### portfolio_comparison_demo.py

**Purpose:** Compare multiple famous portfolio strategies using backtesting

**Description:**
This script demonstrates how to:
- Define multiple portfolio strategies with different asset allocations
- Run backtests for each portfolio
- Compare performance across portfolios
- Generate visualization plots (value history, yield history, fees, and taxes)

**Portfolios Included:**
The script includes 25 famous portfolio strategies:
1. Ideal Index
2. Harry Browne
3. Warren Buffett
4. All Seasons (Ray Dalio)
5. Stocks 100%
6. 2nd Grade
7. No Brainer
8. Three Funds
9. Ivy League
10. Four Cores
11. Margarita
12. Went Fishing
13. 50/50
14. Foursquare
15. Unconventional Success
16. Dark Chocolate
17. Cherry Pie
18. Colliding Markets
19. Ultimate Buy & Hold
20. Coffee Shop
21. Chicken
22. Big Bricks
23. 7-12
24. Thick Tail
25. Talmud

**Usage:**
```bash
python examples/portfolio_comparison_demo.py
```

**Configuration:**
Edit the script to:
- Select which portfolios to compare (modify `portfolios_list`)
- Adjust simulation parameters (liquid, fees, tax, dates)
- Change rebalancing periods
- Enable/disable verbose output and plots

**Output:**
- Portfolio value history plot
- Yield history plot
- Fee and tax history plots

## Creating Your Own Examples

To create a new example:

1. Create a new Python file in this directory
2. Import the necessary modules from `src/`
3. Document the example with a docstring
4. Add it to this README

Example template:
```python
"""
Your Example Name

Description of what this example demonstrates.

Usage:
    python examples/your_example.py
"""

from src.markets import Market
from src.brokers import Broker
from src.traders import Trader

# Your example code here
```

## Notes

- All examples use the refactored codebase with Position objects
- Examples demonstrate real-world usage patterns
- Feel free to modify and experiment with the examples
- For unit tests, see the `tests/` directory instead