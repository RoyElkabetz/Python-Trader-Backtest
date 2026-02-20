# Gymnasium-Style Architecture Refactoring Plan

**Date:** 2026-02-20  
**Goal:** Transform Python Trader Backtest into a Gymnasium-compatible RL environment  
**Status:** Planning Phase

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current vs Target Architecture](#current-vs-target-architecture)
3. [Gymnasium API Overview](#gymnasium-api-overview)
4. [Detailed Design](#detailed-design)
5. [Implementation Plan](#implementation-plan)
6. [Backward Compatibility](#backward-compatibility)
7. [Testing Strategy](#testing-strategy)
8. [Migration Guide](#migration-guide)

---

## Executive Summary

### Objective
Refactor the backtesting framework to follow the **Gymnasium (OpenAI Gym) API standard**, making it compatible with reinforcement learning libraries while maintaining backward compatibility with existing code.

### Key Changes
- **Merge** `Market` and `Broker` classes into unified `TradingEnv` class
- **Implement** standard Gymnasium API: `reset()`, `step()`, `render()`
- **Define** observation and action spaces
- **Refactor** `Trader` to work as an agent (optional policy)
- **Maintain** backward compatibility layer

### Benefits
✅ **RL Compatibility**: Works with Stable-Baselines3, RLlib, Ray  
✅ **Standardization**: Industry-standard API for trading environments  
✅ **Flexibility**: Easy to swap data sources, reward functions, action spaces  
✅ **Testability**: Clear separation of concerns  
✅ **Research**: Enables academic research and experimentation  

---

## Current vs Target Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Simulator                             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │  Market  │◄─────┤  Broker  │◄─────┤  Trader  │          │
│  │          │      │          │      │          │          │
│  │ - data   │      │ - fees   │      │ - buy()  │          │
│  │ - step() │      │ - buy()  │      │ - sell() │          │
│  │          │      │ - sell() │      │ - balance│          │
│  └──────────┘      └──────────┘      └──────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Issues:**
- Tight coupling between components
- No standard interface for RL
- Trader has too many responsibilities
- Hard to extend or modify

### Target Architecture (Gymnasium-Style)

```
┌─────────────────────────────────────────────────────────────┐
│                     TradingEnv (Gymnasium)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Public API                          │   │
│  │  - reset() → observation                            │   │
│  │  - step(action) → obs, reward, done, truncated, info│   │
│  │  - render() → visualization                         │   │
│  │  - observation_space: Box/Dict                      │   │
│  │  - action_space: Box/Discrete/MultiDiscrete         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ MarketData  │  │BrokerExecutor│  │RewardCalc   │        │
│  │ (internal)  │  │  (internal)  │  │ (internal)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ reset(), step(action)
                            │
                    ┌───────┴────────┐
                    │  Agent/Policy  │
                    │  (Trader or RL)│
                    └────────────────┘
```

**Benefits:**
- Clean separation of concerns
- Standard RL interface
- Easy to extend and test
- Pluggable components

---

## Gymnasium API Overview

### Core Methods

#### 1. `reset(seed=None, options=None)`
**Purpose:** Initialize/reset the environment to starting state

**Returns:**
```python
observation: ObsType  # Initial state
info: dict           # Additional information
```

**Example:**
```python
obs, info = env.reset(seed=42)
# obs = {
#     'portfolio': [100, 50, 0],  # units owned per ticker
#     'prices': [150.0, 200.0, 50.0],  # current prices
#     'liquid': 10000.0,  # available cash
#     'date_idx': 0  # current time step
# }
```

#### 2. `step(action)`
**Purpose:** Execute action and advance environment one time step

**Parameters:**
```python
action: ActType  # Action to take (buy/sell/hold)
```

**Returns:**
```python
observation: ObsType  # New state after action
reward: float        # Reward for this step
terminated: bool     # Episode ended naturally (e.g., end of data)
truncated: bool      # Episode ended artificially (e.g., bankruptcy)
info: dict          # Additional information
```

**Example:**
```python
action = [0.3, -0.2, 0.0]  # Buy 30% AAPL, sell 20% GOOG, hold SPY
obs, reward, terminated, truncated, info = env.step(action)
```

#### 3. `render()`
**Purpose:** Visualize current state (optional but recommended)

**Returns:**
```python
None or RenderFrame  # Depends on render_mode
```

### Space Definitions

#### Observation Space
Defines what the agent can observe:

```python
from gymnasium import spaces

observation_space = spaces.Dict({
    'portfolio': spaces.Box(low=0, high=np.inf, shape=(n_tickers,)),
    'prices': spaces.Box(low=0, high=np.inf, shape=(n_tickers,)),
    'liquid': spaces.Box(low=0, high=np.inf, shape=(1,)),
    'portfolio_value': spaces.Box(low=0, high=np.inf, shape=(1,)),
    'date_idx': spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32)
})
```

#### Action Space
Defines what actions the agent can take:

**Option A: Continuous (Box)**
```python
# Actions are portfolio weights [-1, 1] for each ticker
# -1 = sell all, 0 = hold, 1 = buy with all available cash
action_space = spaces.Box(low=-1, high=1, shape=(n_tickers,), dtype=np.float32)
```

**Option B: Discrete**
```python
# Actions are discrete choices per ticker
# 0 = sell, 1 = hold, 2 = buy
action_space = spaces.MultiDiscrete([3] * n_tickers)
```

**Option C: Hybrid (Dict)**
```python
# Separate action type and amount
action_space = spaces.Dict({
    'action_type': spaces.MultiDiscrete([3] * n_tickers),  # sell/hold/buy
    'amount': spaces.Box(low=0, high=1, shape=(n_tickers,))  # percentage
})
```

---

## Detailed Design

### 1. TradingEnv Class Structure

```python
# src/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import date

class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment.
    
    Merges Market and Broker functionality into a unified environment
    that follows the standard RL interface.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4
    }
    
    def __init__(
        self,
        tickers: List[str],
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        initial_liquid: float = 10000.0,
        buy_fee: float = 0.08,
        min_buy_fee: float = 2.0,
        sell_fee: float = 0.08,
        min_sell_fee: float = 2.0,
        tax: float = 25.0,
        benchmark_index: str = '^GSPC',
        render_mode: Optional[str] = None,
        reward_function: str = 'profit',  # 'profit', 'sharpe', 'sortino'
        action_type: str = 'continuous'   # 'continuous', 'discrete', 'hybrid'
    ):
        """
        Initialize trading environment.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date as (year, month, day)
            end_date: End date as (year, month, day)
            initial_liquid: Starting cash amount
            buy_fee: Buy fee percentage
            min_buy_fee: Minimum buy fee
            sell_fee: Sell fee percentage
            min_sell_fee: Minimum sell fee
            tax: Capital gains tax percentage
            benchmark_index: Benchmark index ticker
            render_mode: Rendering mode ('human', 'rgb_array', None)
            reward_function: Reward calculation method
            action_type: Type of action space
        """
        super().__init__()
        
        # Store configuration
        self.tickers = [t.upper() for t in tickers]
        self.n_tickers = len(tickers)
        self.initial_liquid = initial_liquid
        self.render_mode = render_mode
        self.reward_function = reward_function
        
        # Initialize internal components
        self._market_data = MarketData(tickers, start_date, end_date, benchmark_index)
        self._broker = BrokerExecutor(buy_fee, min_buy_fee, sell_fee, min_sell_fee, tax)
        self._reward_calculator = RewardCalculator(reward_function)
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space
        self.action_space = self._create_action_space(action_type)
        
        # State variables (initialized in reset())
        self.current_step = 0
        self.liquid = initial_liquid
        self.portfolio = {}  # {ticker: [Position, ...]}
        self.portfolio_history = []
        self.transaction_history = []
        
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space definition."""
        return spaces.Dict({
            # Portfolio state
            'portfolio_units': spaces.Box(
                low=0, high=np.inf, 
                shape=(self.n_tickers,), 
                dtype=np.float32
            ),
            'portfolio_values': spaces.Box(
                low=0, high=np.inf, 
                shape=(self.n_tickers,), 
                dtype=np.float32
            ),
            
            # Market data
            'prices': spaces.Box(
                low=0, high=np.inf, 
                shape=(self.n_tickers,), 
                dtype=np.float32
            ),
            'price_changes': spaces.Box(
                low=-1, high=1, 
                shape=(self.n_tickers,), 
                dtype=np.float32
            ),
            
            # Account state
            'liquid': spaces.Box(
                low=0, high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            ),
            'total_value': spaces.Box(
                low=0, high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            ),
            
            # Time
            'step': spaces.Box(
                low=0, high=self._market_data.max_steps, 
                shape=(1,), 
                dtype=np.int32
            )
        })
    
    def _create_action_space(self, action_type: str) -> spaces.Space:
        """Create action space definition."""
        if action_type == 'continuous':
            # Continuous actions: portfolio weights [-1, 1]
            return spaces.Box(
                low=-1, high=1, 
                shape=(self.n_tickers,), 
                dtype=np.float32
            )
        elif action_type == 'discrete':
            # Discrete actions: sell/hold/buy for each ticker
            return spaces.MultiDiscrete([3] * self.n_tickers)
        elif action_type == 'hybrid':
            # Hybrid: action type + amount
            return spaces.Dict({
                'action_type': spaces.MultiDiscrete([3] * self.n_tickers),
                'amount': spaces.Box(low=0, high=1, shape=(self.n_tickers,))
            })
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., different start date)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset market data
        self._market_data.reset()
        
        # Reset portfolio state
        self.current_step = 0
        self.liquid = self.initial_liquid
        self.portfolio = {ticker: [] for ticker in self.tickers}
        self.portfolio_history = []
        self.transaction_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'date': self._market_data.current_date,
            'initial_liquid': self.initial_liquid
        }
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action and advance environment.
        
        Args:
            action: Action to take (format depends on action_space)
            
        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Execute action
        self._execute_action(action)
        
        # Advance market one step
        done = self._market_data.step()
        
        # Update portfolio prices
        self._update_portfolio_prices()
        
        # Calculate reward
        current_portfolio_value = self._calculate_portfolio_value()
        reward = self._reward_calculator.calculate(
            prev_value=prev_portfolio_value,
            current_value=current_portfolio_value,
            liquid=self.liquid,
            portfolio_history=self.portfolio_history
        )
        
        # Get new observation
        observation = self._get_observation()
        
        # Check termination conditions
        terminated = done  # End of data
        truncated = self.liquid < 0 or current_portfolio_value <= 0  # Bankruptcy
        
        # Additional info
        info = {
            'date': self._market_data.current_date,
            'portfolio_value': current_portfolio_value,
            'liquid': self.liquid,
            'total_value': current_portfolio_value + self.liquid,
            'step': self.current_step
        }
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Portfolio units
        portfolio_units = np.array([
            sum(pos.units for pos in self.portfolio[ticker])
            for ticker in self.tickers
        ], dtype=np.float32)
        
        # Current prices
        prices = np.array([
            self._market_data.get_price(ticker)
            for ticker in self.tickers
        ], dtype=np.float32)
        
        # Portfolio values
        portfolio_values = portfolio_units * prices
        
        # Price changes (if not first step)
        if self.current_step > 0:
            prev_prices = self.portfolio_history[-1]['prices']
            price_changes = (prices - prev_prices) / prev_prices
        else:
            price_changes = np.zeros(self.n_tickers, dtype=np.float32)
        
        # Total portfolio value
        total_value = np.sum(portfolio_values) + self.liquid
        
        return {
            'portfolio_units': portfolio_units,
            'portfolio_values': portfolio_values,
            'prices': prices,
            'price_changes': price_changes,
            'liquid': np.array([self.liquid], dtype=np.float32),
            'total_value': np.array([total_value], dtype=np.float32),
            'step': np.array([self.current_step], dtype=np.int32)
        }
    
    def _execute_action(self, action: np.ndarray):
        """Execute trading action."""
        # Implementation depends on action space type
        # This is a simplified version for continuous actions
        
        for i, ticker in enumerate(self.tickers):
            action_value = action[i]
            
            if action_value > 0.01:  # Buy
                # Calculate how much to buy
                amount_to_invest = self.liquid * action_value
                price = self._market_data.get_price(ticker)
                units = int(amount_to_invest / price)
                
                if units > 0:
                    self._buy(ticker, units)
                    
            elif action_value < -0.01:  # Sell
                # Calculate how much to sell
                current_units = sum(pos.units for pos in self.portfolio[ticker])
                units_to_sell = int(current_units * abs(action_value))
                
                if units_to_sell > 0:
                    self._sell(ticker, units_to_sell)
    
    def _buy(self, ticker: str, units: int):
        """Execute buy order."""
        position, total_price, fee = self._broker.buy(
            ticker=ticker,
            units=units,
            price=self._market_data.get_price(ticker),
            date=self._market_data.current_date
        )
        
        # Update state
        self.liquid -= (total_price + fee)
        self.portfolio[ticker].append(position)
        
        # Record transaction
        self.transaction_history.append({
            'date': self._market_data.current_date,
            'type': 'BUY',
            'ticker': ticker,
            'units': units,
            'price': position.purchase_price,
            'total': total_price,
            'fee': fee
        })
    
    def _sell(self, ticker: str, units: int):
        """Execute sell order."""
        # Get positions to sell (FIFO by default)
        positions_to_sell = self._get_positions_to_sell(ticker, units)
        
        # Execute sell through broker
        proceeds, fee, tax = self._broker.sell(
            ticker=ticker,
            positions=positions_to_sell,
            price=self._market_data.get_price(ticker)
        )
        
        # Update state
        self.liquid += (proceeds - fee - tax)
        
        # Remove sold positions
        for pos in positions_to_sell:
            self.portfolio[ticker].remove(pos)
        
        # Record transaction
        self.transaction_history.append({
            'date': self._market_data.current_date,
            'type': 'SELL',
            'ticker': ticker,
            'units': units,
            'price': self._market_data.get_price(ticker),
            'total': proceeds,
            'fee': fee,
            'tax': tax
        })
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio market value."""
        total = 0.0
        for ticker in self.tickers:
            price = self._market_data.get_price(ticker)
            units = sum(pos.units for pos in self.portfolio[ticker])
            total += units * price
        return total
    
    def _update_portfolio_prices(self):
        """Update current prices for all positions."""
        for ticker in self.tickers:
            current_price = self._market_data.get_price(ticker)
            for position in self.portfolio[ticker]:
                position.update_price(current_price)
    
    def _render_human(self):
        """Render environment in human-readable format."""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step} | Date: {self._market_data.current_date}")
        print(f"{'='*60}")
        print(f"Liquid: ${self.liquid:,.2f}")
        print(f"\nPortfolio:")
        for ticker in self.tickers:
            units = sum(pos.units for pos in self.portfolio[ticker])
            price = self._market_data.get_price(ticker)
            value = units * price
            print(f"  {ticker}: {units} units @ ${price:.2f} = ${value:,.2f}")
        print(f"\nTotal Value: ${self._calculate_portfolio_value() + self.liquid:,.2f}")
        print(f"{'='*60}\n")
```

### 2. Internal Components

#### MarketData (replaces Market)
```python
# src/env.py (continued)

class MarketData:
    """Internal component for managing market data."""
    
    def __init__(
        self, 
        tickers: List[str], 
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        benchmark_index: str
    ):
        self.tickers = tickers
        self.start_date = date(*start_date)
        self.end_date = date(*end_date)
        self.benchmark_index = benchmark_index
        
        # Load data
        self._load_data()
        
        # State
        self.current_idx = 0
        self.current_date = None
        self.max_steps = len(self.data[tickers[0]])
    
    def _load_data(self):
        """Load market data from yfinance."""
        import yfinance as yf
        
        self.data = {}
        for ticker in self.tickers:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            self.data[ticker] = df
    
    def reset(self):
        """Reset to beginning of data."""
        self.current_idx = 0
        self.current_date = self.data[self.tickers[0]].index[0].date()
    
    def step(self) -> bool:
        """Advance one time step. Returns True if done."""
        self.current_idx += 1
        if self.current_idx >= self.max_steps:
            return True
        self.current_date = self.data[self.tickers[0]].index[self.current_idx].date()
        return False
    
    def get_price(self, ticker: str, price_type: str = 'Open') -> float:
        """Get current price for ticker."""
        return self.data[ticker].iloc[self.current_idx][price_type]
```

#### BrokerExecutor (replaces Broker)
```python
# src/env.py (continued)

class BrokerExecutor:
    """Internal component for executing trades."""
    
    def __init__(
        self,
        buy_fee: float,
        min_buy_fee: float,
        sell_fee: float,
        min_sell_fee: float,
        tax: float
    ):
        self.buy_fee = buy_fee / 100.0
        self.min_buy_fee = min_buy_fee
        self.sell_fee = sell_fee / 100.0
        self.min_sell_fee = min_sell_fee
        self.tax = tax / 100.0
    
    def buy(
        self, 
        ticker: str, 
        units: int, 
        price: float, 
        date: date
    ) -> Tuple[Position, float, float]:
        """Execute buy order."""
        from .position import Position
        
        total_price = price * units
        fee = max(self.buy_fee * total_price, self.min_buy_fee)
        
        position = Position(
            ticker=ticker,
            units=units,
            purchase_price=price,
            purchase_date=date,
            current_price=price
        )
        
        return position, total_price, fee
    
    def sell(
        self,
        ticker: str,
        positions: List[Position],
        price: float
    ) -> Tuple[float, float, float]:
        """Execute sell order."""
        total_units = sum(pos.units for pos in positions)
        proceeds = price * total_units
        
        fee = max(self.sell_fee * proceeds, self.min_sell_fee)
        
        cost_basis = sum(pos.cost_basis for pos in positions)
        tax = max(0, (proceeds - cost_basis) * self.tax)
        
        return proceeds, fee, tax
```

#### RewardCalculator
```python
# src/env.py (continued)

class RewardCalculator:
    """Calculate rewards for RL training."""
    
    def __init__(self, reward_function: str):
        self.reward_function = reward_function
    
    def calculate(
        self,
        prev_value: float,
        current_value: float,
        liquid: float,
        portfolio_history: List[Dict]
    ) -> float:
        """Calculate reward based on configured function."""
        
        if self.reward_function == 'profit':
            # Simple profit-based reward
            return current_value - prev_value
        
        elif self.reward_function == 'return':
            # Percentage return
            if prev_value == 0:
                return 0.0
            return (current_value - prev_value) / prev_value
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio (requires history)
            if len(portfolio_history) < 2:
                return 0.0
            returns = [h['return'] for h in portfolio_history[-30:]]  # Last 30 days
            if len(returns) < 2:
                return 0.0
            return np.mean(returns) / (np.std(returns) + 1e-8)
        
        elif self.reward_function == 'sortino':
            # Sortino ratio (downside deviation)
            if len(portfolio_history) < 2:
                return 0.0
            returns = [h['return'] for h in portfolio_history[-30:]]
            downside = [r for r in returns if r < 0]
            if len(downside) == 0:
                return np.mean(returns)
            return np.mean(returns) / (np.std(downside) + 1e-8)
        
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
```

---

## Implementation Plan

### Phase 1: Core Environment (Week 1)
**Goal:** Create basic TradingEnv with Gymnasium API

- [x] Research Gymnasium API requirements
- [ ] Create `src/env.py` with TradingEnv class
- [ ] Implement MarketData component
- [ ] Implement BrokerExecutor component
- [ ] Implement RewardCalculator component
- [ ] Define observation space
- [ ] Define action space (continuous)
- [ ] Implement `reset()` method
- [ ] Implement `step()` method
- [ ] Basic unit tests

**Deliverables:**
- Working TradingEnv class
- Basic buy/sell/hold functionality
- Simple profit-based rewards

### Phase 2: Enhanced Features (Week 2)
**Goal:** Add advanced features and multiple action spaces

- [ ] Implement discrete action space
- [ ] Implement hybrid action space
- [ ] Add multiple reward functions (Sharpe, Sortino)
- [ ] Implement `render()` method
- [ ] Add observation normalization
- [ ] Add action clipping/validation
- [ ] Integration tests

**Deliverables:**
- Multiple action space options
- Multiple reward functions
- Visualization support

### Phase 3: Backward Compatibility (Week 3)
**Goal:** Maintain compatibility with existing code

- [ ] Create compatibility wrapper for old Trader API
- [ ] Update Trader class to work with TradingEnv
- [ ] Create adapter for old simulator() function
- [ ] Update examples with both APIs
- [ ] Migration guide documentation

**Deliverables:**
- Backward compatibility layer
- Updated examples
- Migration documentation

### Phase 4: RL Integration (Week 4)
**Goal:** Demonstrate RL library integration

- [ ] Create Stable-Baselines3 example
- [ ] Create RLlib example
- [ ] Add pre-trained model examples
- [ ] Performance benchmarks
- [ ] Comprehensive documentation

**Deliverables:**
- Working RL examples
- Pre-trained models
- Performance comparisons

---

## Backward Compatibility

### Strategy
Maintain existing API while adding new Gymnasium interface:

```python
# Old API (still works)
from src.main import simulator

traders = simulator(
    liquid=10000,
    tickers=['AAPL', 'GOOG'],
    periods=[30],
    ratios=[0.5, 0.5],
    sell_strategy='FIFO',
    start_date=(2020, 1, 1),
    end_date=(2020, 12, 31),
    buy_fee=0.08,
    min_buy_fee=2.0,
    sell_fee=0.08,
    min_sell_fee=2.0,
    tax=25.0,
    verbose=True
)

# New API (Gymnasium)
from src.env import TradingEnv

env = TradingEnv(
    tickers=['AAPL', 'GOOG'],
    start_date=(2020, 1, 1),
    end_date=(2020, 12, 31),
    initial_liquid=10000.0,
    buy_fee=0.08,
    min_buy_fee=2.0,
    sell_fee=0.08,
    min_sell_fee=2.0,
    tax=25.0
)

obs, info = env.reset()
done = False
while not done:
    action = agent.predict(obs)  # Your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Compatibility Wrapper
```python
# src/compat.py

class TraderAdapter:
    """Adapter to use old Trader API with new TradingEnv."""
    
    def __init__(self, env: TradingEnv, balance_period: int = 30):
        self.env = env
        self.balance_period = balance_period
        self.step_count = 0
    
    def buy(self, ticker: str, units: int):
        """Old-style buy method."""
        # Convert to action and execute
        action = self._create_buy_action(ticker, units)
        self.env.step(action)
    
    def sell(self, ticker: str, units: int):
        """Old-style sell method."""
        # Convert to action and execute
        action = self._create_sell_action(ticker, units)
        self.env.step(action)
    
    def balance(self, tickers: List[str], p: List[float]):
        """Old-style balance method."""
        # Convert to rebalancing action
        action = self._create_balance_action(tickers, p)
        self.env.step(action)
    
    # ... other compatibility methods
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_env.py

def test_env_creation():
    """Test environment can be created."""
    env = TradingEnv(
        tickers=['AAPL'],
        start_date=(2020, 1, 1),
        end_date=(2020, 12, 31)
    )
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None

def test_reset():
    """Test reset returns valid observation."""
    env = TradingEnv(tickers=['AAPL'], ...)
    obs, info = env.reset()
    
    assert env.observation_space.contains(obs)
    assert 'date' in info
    assert obs['liquid'][0] == env.initial_liquid

def test_step():
    """Test step executes action correctly."""
    env = TradingEnv(tickers=['AAPL'], ...)
    obs, info = env.reset()
    
    action = np.array([0.5])  # Buy 50%
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_buy_action():
    """Test buy action reduces liquid and increases portfolio."""
    env = TradingEnv(tickers=['AAPL'], ...)
    obs, _ = env.reset()
    initial_liquid = obs['liquid'][0]
    
    action = np.array([0.5])  # Buy 50%
    obs, _, _, _, _ = env.step(action)
    
    assert obs['liquid'][0] < initial_liquid
    assert obs['portfolio_units'][0] > 0

def test_sell_action():
    """Test sell action increases liquid and decreases portfolio."""
    env = TradingEnv(tickers=['AAPL'], ...)
    env.reset()
    
    # First buy
    env.step(np.array([0.5]))
    
    # Then sell
    obs_before, _, _, _, _ = env.step(np.array([0.0]))
    obs_after, _, _, _, _ = env.step(np.array([-0.5]))
    
    assert obs_after['liquid'][0] > obs_before['liquid'][0]
    assert obs_after['portfolio_units'][0] < obs_before['portfolio_units'][0]
```

### Integration Tests
```python
def test_full_episode():
    """Test complete episode from reset to done."""
    env = TradingEnv(tickers=['AAPL', 'GOOG'], ...)
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    assert steps > 0
    assert 'total_value' in info

def test_gymnasium_check():
    """Test environment passes Gymnasium's check_env."""
    from gymnasium.utils.env_checker import check_env
    
    env = TradingEnv(tickers=['AAPL'], ...)
    check_env(env)  # Should not raise any errors
```

### RL Integration Tests
```python
def test_stable_baselines3_integration():
    """Test environment works with Stable-Baselines3."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    
    env = TradingEnv(tickers=['AAPL'], ...)
    check_env(env)
    
    # Train for a few steps
    model = PPO('MultiInputPolicy', env, verbose=0)
    model.learn(total_timesteps=100)
    
    # Test prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    assert env.action_space.contains(action)
```

---

## Migration Guide

### For Existing Users

#### Step 1: Install Gymnasium
```bash
pip install gymnasium
```

#### Step 2: Update Imports
```python
# Old
from src.markets import Market
from src.brokers import Broker
from src.traders import Trader

# New (for RL)
from src.env import TradingEnv

# Or keep old imports for backward compatibility
```

#### Step 3: Choose Your Path

**Path A: Keep Using Old API**
```python
# No changes needed! Old code still works
from src.main import simulator

traders = simulator(...)  # Same as before
```

**Path B: Migrate to Gymnasium API**
```python
# Create environment
env = TradingEnv(
    tickers=['AAPL', 'GOOG'],
    start_date=(2020, 1, 1),
    end_date=(2020, 12, 31),
    initial_liquid=10000.0
)

# Use with your own agent
obs, info = env.reset()
done = False
while not done:
    action = your_agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

**Path C: Use with RL Library**
```python
from stable_baselines3 import PPO
from src.env import TradingEnv

# Create environment
env = TradingEnv(...)

# Train RL agent
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Use trained agent
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Key Differences

| Aspect | Old API | New API |
|--------|---------|---------|
| **Entry Point** | `simulator()` function | `TradingEnv` class |
| **Control Flow** | Imperative (trader.buy/sell) | Declarative (env.step) |
| **Agent** | Built-in Trader class | External agent/policy |
| **State** | Internal to Trader | Observation dict |
| **Actions** | Method calls | Numpy arrays |
| **Rewards** | Implicit (profit) | Explicit return value |

---

## Success Metrics

### Functionality
- [ ] Environment passes `gymnasium.utils.env_checker.check_env()`
- [ ] Works with Stable-Baselines3
- [ ] Works with RLlib
- [ ] Backward compatibility maintained
- [ ] All existing tests pass

### Performance
- [ ] Episode runtime < 1 second for 1-year backtest
- [ ] Memory usage < 100MB for typical portfolio
- [ ] Supports parallel environments

### Code Quality
- [ ] 90%+ test coverage
- [ ] Type hints on all public methods
- [ ] Comprehensive documentation
- [ ] Example notebooks

---

## Next Steps

1. **Review this plan** and provide feedback
2. **Prioritize features** (which action spaces, reward functions?)
3. **Set timeline** (aggressive 4 weeks or relaxed 8 weeks?)
4. **Begin Phase 1** implementation
5. **Iterate** based on testing and feedback

---

## Questions for Discussion

1. **Action Space**: Which action space should be the default? Continuous, discrete, or hybrid?
2. **Reward Function**: Which reward function is most important? Profit, Sharpe, Sortino, or custom?
3. **Observation Space**: Should we include technical indicators (RSI, MACD) in observations?
4. **Backward Compatibility**: How important is maintaining 100% backward compatibility?
5. **RL Libraries**: Which RL libraries should we prioritize? (Stable-Baselines3, RLlib, others?)
6. **Performance**: Any specific performance requirements or constraints?

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-20  
**Author:** IBM Bob (AI Assistant)  
**Status:** Ready for Review