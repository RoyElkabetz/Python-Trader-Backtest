import argparse
from .simulators import multi_period_simulator
from .logging_config import setup_logging
import copy as cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtesting Trading Simulator')

    # Trading Arguments
    parser.add_argument('--tickers', type=str, required=True, nargs='+',
                        help='List of stock tickers to trade')
    parser.add_argument('--periods', type=int, required=True, nargs='+',
                        help='Balance periods to test (in days)')
    parser.add_argument('--ratios', type=float, required=True, nargs='+',
                        help='Target portfolio allocation ratios')
    parser.add_argument('--start-date', type=int, required=True, nargs='+',
                        help='Start date as YYYY MM DD')
    parser.add_argument('--end-date', type=int, required=True, nargs='+',
                        help='End date as YYYY MM DD')
    parser.add_argument('--deposit', type=float, default=0.0,
                        help='Periodic deposit amount')
    parser.add_argument('--deposit-period', type=int, default=30,
                        help='Days between deposits')
    parser.add_argument('--buy-fee', type=float, default=0.08,
                        help='Buy transaction fee percentage')
    parser.add_argument('--min-buy-fee', type=float, default=2.,
                        help='Minimum buy transaction fee')
    parser.add_argument('--sell-fee', type=float, default=0.08,
                        help='Sell transaction fee percentage')
    parser.add_argument('--min-sell-fee', type=float, default=2.,
                        help='Minimum sell transaction fee')
    parser.add_argument('--tax', type=float, default=25.,
                        help='Capital gains tax percentage')
    parser.add_argument('--liquid', type=float, required=True,
                        help='Initial liquid cash')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Print detailed trading information')
    parser.add_argument('--sell-strategy', type=str, default='FIFO',
                        choices=['FIFO', 'LIFO', 'TAX_OPT'],
                        help='Strategy for selling stocks')
    
    # Logging Arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Console log level (default: INFO)')
    parser.add_argument('--file-log-level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='File log level (default: DEBUG)')
    parser.add_argument('--log-file', type=str, default='trader_backtest.log',
                        help='Log file name (default: trader_backtest.log)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Log directory (default: logs/)')
    parser.add_argument('--no-console-log', action='store_true',
                        help='Disable console logging')
    parser.add_argument('--no-file-log', action='store_true',
                        help='Disable file logging')
    
    args = parser.parse_args()
    
    # Initialize logging system
    setup_logging(
        console_level=args.log_level,
        file_level=args.file_log_level,
        log_file=args.log_file,
        log_dir=args.log_dir,
        enable_console=not args.no_console_log,
        enable_file=not args.no_file_log,
    )

    # run simulation
    multi_period_simulator(
        args.liquid, 
        args.tickers, 
        args.periods, 
        args.ratios, 
        args.sell_strategy, 
        args.start_date, 
        args.end_date,
        args.buy_fee, 
        args.min_buy_fee, 
        args.sell_fee, 
        args.min_sell_fee, 
        args.tax, 
        args.verbose,
        args.deposit, 
        args.deposit_period
        )
