"""
Tests for the logging system.
"""

import pytest
import os
from pathlib import Path
import tempfile
import shutil
from src.logging_config import LoggerSetup, setup_logging, get_logger
from src.markets import Market
from src.brokers import Broker
from src.traders import Trader
from src.simulators import base_simulator


class TestLoggingConfiguration:
    """Test suite for logging configuration."""
    
    def test_logger_setup_initialization(self):
        """Test that logger setup initializes correctly."""
        LoggerSetup.reset()
        logger = setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            enable_console=True,
            enable_file=False
        )
        assert logger is not None
        assert logger.name == "trader_backtest"
    
    def test_get_logger_creates_hierarchical_logger(self):
        """Test that get_logger creates hierarchical loggers."""
        LoggerSetup.reset()
        setup_logging(enable_console=False, enable_file=False)
        
        logger = get_logger("markets")
        assert logger.name == "trader_backtest.markets"
        
        logger2 = get_logger("brokers")
        assert logger2.name == "trader_backtest.brokers"
    
    def test_logger_setup_with_file_logging(self):
        """Test logger setup with file logging enabled."""
        # Use temporary directory for test logs
        with tempfile.TemporaryDirectory() as tmpdir:
            LoggerSetup.reset()
            test_log_file = "test_log.log"
            
            logger = setup_logging(
                console_level="INFO",
                file_level="DEBUG",
                log_dir=tmpdir,
                log_file=test_log_file,
                enable_console=False,
                enable_file=True
            )
            
            # Log a test message
            logger.info("Test message")
            
            # Check that log file was created
            log_path = Path(tmpdir) / test_log_file
            assert log_path.exists()
            
            # Check that message was written
            with open(log_path, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_logger_setup_console_only(self):
        """Test logger setup with console logging only."""
        LoggerSetup.reset()
        logger = setup_logging(
            console_level="INFO",
            enable_console=True,
            enable_file=False
        )
        
        # Should have one handler (console)
        assert len(logger.handlers) == 1
    
    def test_logger_setup_file_only(self):
        """Test logger setup with file logging only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            LoggerSetup.reset()
            logger = setup_logging(
                file_level="DEBUG",
                log_dir=tmpdir,
                log_file="test.log",
                enable_console=False,
                enable_file=True
            )
            
            # Should have one handler (file)
            assert len(logger.handlers) == 1
    
    def test_logger_setup_both_handlers(self):
        """Test logger setup with both console and file logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            LoggerSetup.reset()
            logger = setup_logging(
                console_level="INFO",
                file_level="DEBUG",
                log_dir=tmpdir,
                log_file="test.log",
                enable_console=True,
                enable_file=True
            )
            
            # Should have two handlers (console + file)
            assert len(logger.handlers) == 2
    
    def test_logger_reset(self):
        """Test that logger reset clears handlers."""
        LoggerSetup.reset()
        setup_logging(enable_console=True, enable_file=False)
        
        LoggerSetup.reset()
        
        # After reset, should be able to set up again
        logger = setup_logging(enable_console=True, enable_file=False)
        assert logger is not None


class TestLoggingIntegration:
    """Integration tests for logging across components."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    def test_market_logging(self, temp_log_dir):
        """Test that Market class logs correctly."""
        LoggerSetup.reset()
        setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            log_dir=temp_log_dir,
            log_file="market_test.log",
            enable_console=False,
            enable_file=True
        )
        
        # Create market (should log initialization)
        market = Market(['AAPL'], (2023, 1, 1), (2023, 2, 1))
        
        # Check log file
        log_path = Path(temp_log_dir) / "market_test.log"
        assert log_path.exists()
        
        with open(log_path, 'r') as f:
            content = f.read()
            assert "Initializing Market" in content
            assert "AAPL" in content
    
    def test_broker_logging(self, temp_log_dir):
        """Test that Broker class logs correctly."""
        LoggerSetup.reset()
        setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            log_dir=temp_log_dir,
            log_file="broker_test.log",
            enable_console=False,
            enable_file=True
        )
        
        market = Market(['AAPL'], (2023, 1, 1), (2023, 2, 1))
        broker = Broker(0.08, 2.0, 0.08, 2.0, 25.0, market)
        
        # Check log file
        log_path = Path(temp_log_dir) / "broker_test.log"
        with open(log_path, 'r') as f:
            content = f.read()
            assert "Broker initialized" in content
            assert "buy_fee" in content
    
    def test_trader_logging(self, temp_log_dir):
        """Test that Trader class logs correctly."""
        LoggerSetup.reset()
        setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            log_dir=temp_log_dir,
            log_file="trader_test.log",
            enable_console=False,
            enable_file=True
        )
        
        market = Market(['AAPL'], (2023, 1, 1), (2023, 2, 1))
        broker = Broker(0.08, 2.0, 0.08, 2.0, 25.0, market)
        trader = Trader(
            liquid=10000.0,
            balance_period=30,
            ratios=[1.0],
            deposit=0.0,
            deposit_period=30,
            broker=broker,
            market=market,
            verbose=False
        )
        
        # Check log file
        log_path = Path(temp_log_dir) / "trader_test.log"
        with open(log_path, 'r') as f:
            content = f.read()
            assert "Trader initialized" in content
            assert "liquid" in content
    
    def test_simulator_logging(self, temp_log_dir):
        """Test that simulator functions log correctly."""
        LoggerSetup.reset()
        setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            log_dir=temp_log_dir,
            log_file="simulator_test.log",
            enable_console=False,
            enable_file=True
        )
        
        market = Market(['AAPL'], (2023, 1, 1), (2023, 1, 31))
        broker = Broker(0.08, 2.0, 0.08, 2.0, 25.0, market)
        trader = Trader(
            liquid=10000.0,
            balance_period=10,
            ratios=[1.0],
            deposit=0.0,
            deposit_period=30,
            broker=broker,
            market=market,
            verbose=False
        )
        
        # Run simulation
        trader, broker, market = base_simulator(
            market=market,
            broker=broker,
            trader=trader,
            verbose=False
        )
        
        # Check log file
        log_path = Path(temp_log_dir) / "simulator_test.log"
        with open(log_path, 'r') as f:
            content = f.read()
            assert "Starting base simulator" in content
            assert "Simulation complete" in content
    
    def test_complete_workflow_logging(self, temp_log_dir):
        """Test logging throughout a complete simulation workflow."""
        LoggerSetup.reset()
        setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            log_dir=temp_log_dir,
            log_file="workflow_test.log",
            enable_console=False,
            enable_file=True
        )
        
        # Run complete workflow
        market = Market(['AAPL', 'MSFT'], (2023, 1, 1), (2023, 2, 28))
        broker = Broker(0.08, 2.0, 0.08, 2.0, 25.0, market)
        trader = Trader(
            liquid=10000.0,
            balance_period=15,
            ratios=[0.5, 0.5],
            deposit=0.0,
            deposit_period=30,
            broker=broker,
            market=market,
            verbose=False
        )
        
        trader, broker, market = base_simulator(
            market=market,
            broker=broker,
            trader=trader,
            verbose=False
        )
        
        # Check log file contains entries from all components
        log_path = Path(temp_log_dir) / "workflow_test.log"
        with open(log_path, 'r') as f:
            content = f.read()
            
            # Market logs
            assert "trader_backtest.markets" in content
            assert "Initializing Market" in content
            
            # Broker logs
            assert "trader_backtest.brokers" in content
            assert "Broker initialized" in content
            
            # Trader logs
            assert "trader_backtest.traders" in content
            assert "Trader initialized" in content
            
            # Simulator logs
            assert "trader_backtest.simulators" in content
            assert "Starting base simulator" in content
            assert "Simulation complete" in content
            
            # Trading operations
            assert "Buy" in content or "buy" in content
            assert "balance" in content or "rebalancing" in content


class TestLoggingLevels:
    """Test different logging levels."""
    
    def test_debug_level_captures_all(self):
        """Test that DEBUG level captures all messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            LoggerSetup.reset()
            logger = setup_logging(
                file_level="DEBUG",
                log_dir=tmpdir,
                log_file="debug_test.log",
                enable_console=False,
                enable_file=True
            )
            
            test_logger = get_logger("test")
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            
            log_path = Path(tmpdir) / "debug_test.log"
            with open(log_path, 'r') as f:
                content = f.read()
                assert "Debug message" in content
                assert "Info message" in content
                assert "Warning message" in content
    
    def test_info_level_filters_debug(self):
        """Test that INFO level filters out DEBUG messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            LoggerSetup.reset()
            logger = setup_logging(
                file_level="INFO",
                log_dir=tmpdir,
                log_file="info_test.log",
                enable_console=False,
                enable_file=True
            )
            
            test_logger = get_logger("test")
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            
            log_path = Path(tmpdir) / "info_test.log"
            with open(log_path, 'r') as f:
                content = f.read()
                assert "Debug message" not in content
                assert "Info message" in content
                assert "Warning message" in content


# Made with Bob