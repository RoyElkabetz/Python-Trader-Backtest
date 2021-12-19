import sys
import numpy as np
import copy as cp
from datetime import date
import PySimpleGUI as sg
sys.path.insert(1, '/Users/royelkabetz/Git/Stock_Trade_Simulator/src')

from src.markets import Market
from src.brokers import Broker
from src.traders import Trader
from src.utils import plot_trader, compare_traders, plot_market


# default arguments
APP_WIDTH = 90
LIEN_WIDTH = 150
TEXT_FONT = "Helvetica"
TEXT_SIZE = 12
TEXT_BOX_SIZE = 10
UNITS_BOX_SIZE = 5
TEXT_HEAD_SIZE = 14
HEADING_SIZE = 16
PROGRESS_BAR_UNITS = 100000

# colors
BLUE_BUTTON_COLOR = '#FFFFFF on #2196f2'
RED_BUTTON_COLOR = '#FFFFFF on #fa5f55'
GREEN_BUTTON_COLOR = '#FFFFFF on #00c851'
SEPARATOR_COLOR = '#FFFFFF'
LIGHT_GRAY_BUTTON_COLOR = f'#212021 on #e0e0e0'
DARK_GRAY_BUTTON_COLOR = '#e0e0e0 on #212021'


def make_gui(theme):
    sg.theme(theme)

    # Market layout
    market_layout = [[sg.Text('Market', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Dates:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('(2019, 1, 1) - (2021, 2, 15)',
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
                      sg.Text('Deposit period:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(None, size=TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-DEPOSIT-PERIOD-'),
                      sg.Text('(days)', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),],
                     [sg.Text('Withdraw:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(0, size=2 * TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-WITHDRAW-'),
                      sg.Text('$', size=UNITS_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Text('Withdraw period:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input(None, size=TEXT_BOX_SIZE,
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
                     [sg.ProgressBar(PROGRESS_BAR_UNITS, orientation='h', size=(80, 20), bar_color=('green', 'white'),
                                     key='-PROGRESS BAR-')]]
    input_layout += [[sg.Button('RUN', size=(77, 2), button_color=GREEN_BUTTON_COLOR, k='-RUN-'),
                      sg.Button('HELP', size=(11, 2), k='-HELP-'),
                      sg.Button('EXIT', size=(12, 2), k='-EXIT-')]]

    # Logger layout
    logging_layout = [[sg.Text('Logger', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                      [sg.Output(size=(140, 50), font='Courier 8')]]

    # Plots layout
    graphing_layout = [[sg.Text('Plots', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                       [sg.Canvas(size=(40, 10), key='-PLOTS-')]]

    # Theming layout
    theme_layout = [[sg.Text("See how elements look under different themes by choosing a different theme here!")],
                    [sg.Listbox(values=sg.theme_list(),
                                size=(20, 12),
                                key='-THEME LISTBOX-',
                                enable_events=True)],
                    [sg.Button("Set Theme")]]

    # Main layout
    layout = [[sg.Text('BackTesting Trading Simulation Application', size=(81, 1), justification='center',
                       font=(TEXT_FONT, HEADING_SIZE), k='-TEXT HEADING-', enable_events=True)]]

    layout += [[sg.TabGroup([[sg.Tab('Inputs', input_layout),
                              sg.Tab('Plots', graphing_layout),
                              sg.Tab('Theming', theme_layout),
                              sg.Tab('Output', logging_layout)]], key='-TAB GROUP-')]]

    return sg.Window('BackTesting Trading Simulation Application', layout)


def run_gui():
    window = make_gui(sg.theme())

    # This is an Event Loop
    while True:
        event, values = window.read()
        if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
            # Print everything to the logger
            print('============ Event = ', event, ' ==============')
            print('-------- Values Dictionary (key=value) --------')
            for key in values:
                print(key, ' = ', values[key])
        if event in (None, '-EXIT-'):
            # Exit program
            print("[LOG] Clicked Exit!")
            break
        elif event == '-HELP-':
            # Help Popup window
            sg.popup('This Gui is a Backtesting Trading simulator. It is used for simulating '
                     'a trading balancing strategy on a given set of stocks (Tickers).')

        elif event == '-RUN-':
            progress_bar = window['-PROGRESS BAR-']

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
            deposit_period = np.int(values['-DEPOSIT-PERIOD-']) if values['-DEPOSIT-PERIOD-'] is not None else None
            withdraw_amount = np.float(values['-DEPOSIT-'])
            withdraw_period = np.int(values['-WITHDRAW-PERIOD-']) if values['-WITHDRAW-PERIOD-'] is not None else None
            tickers = [ticker.strip() for ticker in values['-TICKERS-'].split(',')]
            ratios = [np.float(ratio.strip()) for ratio in values['-RATIOS-'].split(',')]
            periods = [np.int(period.strip()) for period in values['-PERIODS-'].split(',')]
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
            if not deposit_amount >= 0:
                sg.popup('Deposit amount should be positive float.')
            if not withdraw_amount >= 0:
                sg.popup('Withdraw amount should be positive float.')
            if not deposit_period > 0:
                sg.popup('Deposit period should be a positive integer.')
            if not withdraw_period > 0:
                sg.popup('Withdraw period should be a positive integer.')

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
                    if steps % deposit_period == 0:
                        trader.deposit(deposit_amount)
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

                traders_list.append(trader)

            print("[LOG] The simulation is complete")
        elif event == "-PLOTS-":
            pass
            # # plot results
            # plot_market(market, normalize=plots_normalize)
            # compare_traders(traders_list, periods, 'bp', interval=np.int(len(trader.date_history) / 10))

            # graph = window['-GRAPH-']  # type: sg.Graph
            # graph.draw_circle(values['-GRAPH-'], fill_color='yellow', radius=20)
            # print("[LOG] Circle drawn at: " + str(values['-GRAPH-']))
        elif event == "Set Theme":
            print("[LOG] Clicked Set Theme!")
            theme_chosen = values['-THEME LISTBOX-'][0]
            print("[LOG] User Chose Theme: " + str(theme_chosen))
            window.close()
            window = make_gui(theme_chosen)

    window.close()
    exit(0)


if __name__ == '__main__':
    run_gui()
