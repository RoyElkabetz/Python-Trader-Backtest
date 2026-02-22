import argparse
from .simulators import multi_period_simulator
import copy as cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtesting Trading Simulator')

    # Arguments
    parser.add_argument('--tickers', type=str, required=True, nargs='+')
    parser.add_argument('--periods', type=int, required=True, nargs='+')
    parser.add_argument('--ratios', type=float, required=True, nargs='+')
    parser.add_argument('--start-date', type=int, required=True, nargs='+')
    parser.add_argument('--end-date', type=int, required=True, nargs='+')
    parser.add_argument('--deposit', type=float, default=0.0)
    parser.add_argument('--deposit-period', type=int, default=30)
    parser.add_argument('--buy-fee', type=float, default=0.08)
    parser.add_argument('--min-buy-fee', type=float, default=2.)
    parser.add_argument('--sell-fee', type=float, default=0.08)
    parser.add_argument('--min-sell-fee', type=float, default=2.)
    parser.add_argument('--tax', type=float, default=25.)
    parser.add_argument('--liquid', type=float, required=True)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--sell-strategy', type=str, default='FIFO', choices=['FIFO', 'LIFO', 'TAX_OPT'])
    args = parser.parse_args()

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
