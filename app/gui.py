import sys
import numpy as np
import copy as cp
from datetime import date
import PySimpleGUI as sg
sys.path.insert(1, '../src')

from src.markets import Market
from src.brokers import Broker
from src.traders import Trader
from utils import market_plot, profit_and_portfolio_value, profits, portfolio_values, liquids, fees_and_tax, yields
from utils import clean_string, delete_figure_agg, draw_figure, yields_usd

# default arguments
APP_WIDTH = 110
LIEN_WIDTH = 200
TEXT_FONT = "Helvetica"
TEXT_SIZE = 12
TEXT_BOX_SIZE = 10
UNITS_BOX_SIZE = 5
TEXT_HEAD_SIZE = 14
HEADING_SIZE = 16
PROGRESS_BAR_UNITS = 1000000
figure_w, figure_h = 700, 300

# colors
BLUE_BUTTON_COLOR = '#FFFFFF on #2196f2'
RED_BUTTON_COLOR = '#FFFFFF on #fa5f55'
GREEN_BUTTON_COLOR = '#FFFFFF on #00c851'
SEPARATOR_COLOR = '#FFFFFF'
LIGHT_GRAY_BUTTON_COLOR = f'#212021 on #e0e0e0'
DARK_GRAY_BUTTON_COLOR = '#e0e0e0 on #212021'
PROGRESS_BAR_COLOR = '#F08080'

fig_dict = {' market plot': market_plot,
            ' profit and value': profit_and_portfolio_value,
            ' profits': profits,
            ' portfolio values': portfolio_values,
            ' liquids': liquids,
            ' fees and tax': fees_and_tax,
            ' yields': yields,
            ' yields usd': yields_usd}


def make_gui(theme):
    sg.theme(theme)

    # Market layout
    market_layout = [[sg.Text('Market', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Dates:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('(2021, 1, 1) - (2021, 2, 15)',
                               key='-DATES-',
                               size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left',
                               font=(TEXT_FONT, TEXT_SIZE))],
                     [sg.HSeparator(color=SEPARATOR_COLOR)],]

    # Broker layout
    broker_layout = [[sg.Text('Broker', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Buy fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(0.08, size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-BUY-'),
                      sg.Text('%', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Sell fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(0.08, size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-SELL-'),
                      sg.Text('%', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Tax:', size=4, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(25, size=TEXT_BOX_SIZE-2, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-TAX-'),
                      sg.Text('%', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),],
                     [sg.Text('Min buy fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(2, size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-MIN-BUY-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Min sell fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(2, size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-MIN-SELL-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      ],
                     [sg.HSeparator(color=SEPARATOR_COLOR)], ]

    # Trader layout
    trader_layout = [[sg.Text('Trader', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Liquid:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(100000, size=2*TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-LIQUID-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE))],
                     [sg.Text('Deposit:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(0, size=2*TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-DEPOSIT-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Deposit period:', size=TEXT_BOX_SIZE + 4, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(size=TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-DEPOSIT-PERIOD-'),
                      sg.Text('(days)', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),],
                     [sg.Text('Withdraw:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(0, size=2 * TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-WITHDRAW-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Withdraw period:', size=TEXT_BOX_SIZE + 4, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(size=TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-WITHDRAW-PERIOD-'),
                      sg.Text('(days)', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)), ],
                     [sg.Text('Tickers:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('AAPL, GOOG, TSLA, ORCL', size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-TICKERS-')],
                     [sg.Text('Ratios:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('0.25, 0.25, 0.25, 0.25', size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-RATIOS-')],
                     [sg.Text('Periods:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('2, 4, 8, 16, 32', size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-PERIODS-')],
                     [sg.Text('Sell Strategy:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.DropDown(list(['FIFO', 'LIFO', 'TAX_OPT']), default_value='FIFO', size=(15, 10), enable_events=False,
                                  font=(TEXT_FONT, TEXT_SIZE), key='-SELL STRATEGY-')],
                     [sg.Text('Verbose:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Checkbox('', default=False, key='-VERBOSE-')],
                     [sg.HSeparator(color=SEPARATOR_COLOR)],]

    # Input layout
    input_layout = [[sg.HSeparator(color=SEPARATOR_COLOR)]]
    input_layout += market_layout + broker_layout + trader_layout
    input_layout += [[sg.Text('Progress Bar', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE),
                              size=APP_WIDTH)],
                     [sg.ProgressBar(PROGRESS_BAR_UNITS, orientation='h', size=(APP_WIDTH - TEXT_BOX_SIZE - 2, 20),
                                     bar_color=(PROGRESS_BAR_COLOR, 'white'), key='-PROGRESS BAR-')]]
    input_layout += [[sg.Button('RUN', size=(77, 2), button_color=GREEN_BUTTON_COLOR, k='-RUN-'),
                      sg.Button('HELP', size=(25, 2), k='-HELP-'),
                      sg.Button('EXIT', size=(25, 2), k='-EXIT-')]]

    # Logger layout
    logging_layout = [[sg.Text('Logger', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                      [sg.Output(size=(140, 50), font='Courier 8')]]

    # Plots layout
    listbox_values = list(fig_dict)
    col_listbox = sg.Col([[sg.Listbox(values=listbox_values, change_submits=True, size=(20, len(listbox_values)),
                               key='-LISTBOX-')],
                          [sg.Button('PLOT', size=(20, 1), button_color=BLUE_BUTTON_COLOR, k='-PLOT-')],
                          [sg.Button('INFO', size=(20, 1), button_color=GREEN_BUTTON_COLOR, k='-INFO-')]])
    col_plot = sg.Pane([sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='-CANVAS-')]])], size=(figure_w + 10, figure_h + 100))
    plotting_layout = [[sg.Text('Plots', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                       [col_listbox, col_plot]]

    # Main layout
    layout = [[sg.Text('BackTesting Trading Simulation Application', size=(APP_WIDTH - TEXT_BOX_SIZE, 1),
                       justification='center', font=(TEXT_FONT, HEADING_SIZE), k='-TEXT HEADING-', enable_events=True)]]

    layout += [[sg.TabGroup([[sg.Tab('Inputs', input_layout),
                              sg.Tab('Plots', plotting_layout),
                              sg.Tab('Output', logging_layout)]], key='-TAB GROUP-')]]

    return sg.Window('BackTesting Trading Simulation Application', layout)


def run_gui():
    window = make_gui('BlueMono')
    canvas_elem = window['-CANVAS-']
    progress_bar = window['-PROGRESS BAR-']
    figure_agg = None

    # This is an Event Loop
    while True:
        event, values = window.read()

        if event in (None, '-EXIT-'):
            # Exit program
            print("[LOG] Clicked Exit!")
            break
        elif event == '-HELP-':
            # Help Popup window
            sg.popup('This Gui is a Backtesting Trading simulator. It is used for simulating '
                     'a trading balancing strategy on a given set of stocks (Tickers).')
        elif event == '-RUN-':

            # Extract arguments from variables
            dates = values['-DATES-'].split('-')
            start_date = tuple(np.array(dates[0].strip()[1:-1].split(','), dtype=np.int))
            end_date = tuple(np.array(dates[1].strip()[1:-1].split(','), dtype=np.int))
            buy_fee = np.float(values['-BUY-'])
            min_buy_fee = np.float(values['-MIN-BUY-'])
            sell_fee = np.float(values['-SELL-'])
            min_sell_fee = np.float(values['-MIN-SELL-'])
            tax = np.float(values['-TAX-'])
            liquid = np.float(values['-LIQUID-'])
            deposit_amount = np.float(values['-DEPOSIT-'])
            deposit_period = None if values['-DEPOSIT-PERIOD-'] == '' else np.int(values['-DEPOSIT-PERIOD-'])
            withdraw_amount = np.float(values['-DEPOSIT-'])
            withdraw_period = None if values['-WITHDRAW-PERIOD-'] == '' else np.int(values['-WITHDRAW-PERIOD-'])
            tickers = clean_string(values['-TICKERS-'], 'letter')
            ratios = clean_string(values['-RATIOS-'], 'float')
            periods = clean_string(values['-PERIODS-'], 'int')
            sell_strategy = values['-SELL STRATEGY-']
            verbose = values['-VERBOSE-']

            # Verify input values are in bounds
            if not np.sum(ratios) == 1:
                sg.popup('Ratios should sum up to 1')
                continue
            if not 0 <= tax <= 100:
                sg.popup('Tax should be between 0 and 100')
                continue
            if not 0 <= buy_fee <= 100:
                sg.popup('Buy fee should be between 0 and 100')
                continue
            if not 0 <= sell_fee <= 100:
                sg.popup('Sell fee should be between 0 and 100')
                continue
            if not date(*start_date) < date(*end_date):
                sg.popup('Start date should be smaller than end date')
                continue
            if not min_buy_fee >= 0 and min_sell_fee >= 0:
                sg.popup('Minimum sell and buy fees should be positive')
                continue
            if deposit_amount < 0:
                sg.popup('Deposit amount should be positive float.')
                continue
            if withdraw_amount < 0:
                sg.popup('Withdraw amount should be positive float.')
                continue
            if deposit_period is not None and not deposit_period > 0:
                sg.popup('Deposit period should be a positive integer.')
                continue
            if withdraw_period is not None and not withdraw_period > 0:
                sg.popup('Withdraw period should be a positive integer.')
                continue

            # Running the simulation
            print("[LOG] Starting BackTesting Simulation")
            traders_list = []
            market = Market(tickers, start_date=start_date, end_date=end_date)
            broker = Broker(buy_fee=buy_fee, min_buy_fee=min_buy_fee, sell_fee=sell_fee,
                            min_sell_fee=min_sell_fee, tax=tax, my_market=market)
            first_date = cp.copy(market.current_date)
            amounts_withdrawn = []

            # progress bar units
            steps_to_finish = market.steps * len(periods)
            cntr_unit = PROGRESS_BAR_UNITS // steps_to_finish
            cntr = 0

            # Iterating through periods
            for i, period in enumerate(periods):
                print(f'period: {period}')

                # Init market
                market.current_idx = 0
                market.current_date = first_date

                # Init new trader
                trader = Trader(liquid=liquid, balance_period=period, broker=broker, market=market,
                                verbose=verbose, sell_strategy=sell_strategy)

                # Init portfolio by buying a single stock from each ticker
                for ticker in tickers:
                    trader.buy(ticker, 1)
                trader.balance(tickers, p=ratios)

                # Iterating through trading days
                done = False
                steps = 0
                while not done:
                    progress_bar.UpdateBar(cntr)
                    cntr += cntr_unit
                    steps += 1
                    if steps % 100 == 0:
                        print('| Step: {:6.0f} / {:6.0f} | Balance period: {:4.0f} |'
                              .format(steps, market.steps, trader.balance_period))

                    # Deposit and Withdraw money
                    if deposit_period is not None:
                        if steps % deposit_period == 0:
                            trader.deposit(deposit_amount)
                    if withdraw_period is not None:
                        if steps % withdraw_period == 0:
                            amount = trader.withdraw(withdraw_amount)
                            amounts_withdrawn.append(amount)

                    # Step market forward in time
                    done, previous_date = market.step()

                    # Step trader forward in time
                    trader.step(previous_date)

                    # Balance trader portfolio in period
                    if steps % trader.balance_period == 0:
                        trader.balance(tickers, p=ratios)

                progress_bar.UpdateBar(cntr)
                traders_list.append(trader)

            progress_bar.UpdateBar(cntr)
            run_flag = True
            print("[LOG] The simulation is complete")
        elif event == "-PLOT-":

            if figure_agg:
                # ** IMPORTANT ** Clean up previous drawing before drawing again
                delete_figure_agg(figure_agg)
            # get first listbox item chosen (returned as a list)
            if len(values['-LISTBOX-']) == 0:
                sg.popup('Please pick a plot from the list')
                continue
            choice = values['-LISTBOX-'][0]

            # call function to get the figure
            if choice == ' market plot':
                if 'run_flag' in locals():
                    fig = market_plot(market, normalize=True)
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' profit and value':
                if 'run_flag' in locals():
                    fig = profit_and_portfolio_value(traders_list, periods, 'period')
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' profits':
                if 'run_flag' in locals():
                    fig = profits(traders_list, periods, 'period')
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' portfolio values':
                if 'run_flag' in locals():
                    fig = portfolio_values(traders_list, periods, 'period')
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' liquids':
                if 'run_flag' in locals():
                    fig = liquids(traders_list, periods, 'period')
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' fees and tax':
                if 'run_flag' in locals():
                    fig = fees_and_tax(traders_list, periods, 'period')
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' yields':
                if 'run_flag' in locals():
                    fig = yields(traders_list, periods, 'period', market)
                else:
                    sg.popup('Please run the simulation first')
                    continue
            if choice == ' yields usd':
                if 'run_flag' in locals():
                    fig = yields_usd(traders_list, periods, 'period', market, liquid)
                else:
                    sg.popup('Please run the simulation first')
                    continue

            # draw the figure
            figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
        elif event == '-INFO-':
            sg.popup("Plots Explanation:",
                     "market plot: plot the 'Open' value for all tickers chosen as a function of time.",
                     "profit: A trader's profit is defined as the TMV - VWB - TAF.",
                     "TMV = Total market value (of the portfolio) at the current time.",
                     "VWB = Value when bought (of all the stocks in the portfolio) at the current time.",
                     "TAF = total Tax and Fees paid by the trader up until the current moment.",
                     "portfolio values: Plot of the portfolio market value as a function of time.",
                     "liquids: A plot of the trader's liquid as a function of time.",
                     "fees and tax: A plot of the total amount of fees and tax paid by the trader ",
                     "as a function of operation number.")

    window.close()
    exit(0)


if __name__ == '__main__':
    run_gui()
