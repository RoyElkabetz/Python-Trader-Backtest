#
#     _________  _________       ____       _________  _________
#    /         |/         |     /    \     /         |/         |
#    $$$$$$$$$/ $$$$$$$$$  |    $$$$  \    $$$$$$$$$/ $$$$$$$$$/
#    $$ | /    |$$      $$/    $$  $$  \   $$      |  $$      |
#    $$ |_$$$$ |$$$$$$$$$ \   $$$$$$$$  \  $$$$$$$/   $$$$$$$/
#    $$     $$ |$$ |   $$  \ $$/     $$  \ $$ |       $$ |
#    $$$$$$$$$/ $$/      $$/$$/        $$/ $$/        $$/
#     _________  __         __     __      ____      _________   _________  __________  __________
#    /         |/  |       /  |   /  |    /    \    /         | /         |/          |/          |
#    $$$$$$$$$/ $$ |       $$ |___$$ /    $$$$  \   $$$$$$$$$  |$$$$$$$$$/ $$$$$$$$$$/ $$$$$$$$$$/      ____
#    $$ /   |   $$ |       $$   $$  /    $$  $$  \  $$ /    $$/ $$ /   |       $$ |          $$        /    \
#    $$$$$$/___ $$ |______ $$$$$    \   $$$$$$$$  \ $$$$$$$$$ \ $$$$$$/___     $$ |        $$_____     $$$$  |
#    $$ /      |$$        |$$ | $$   | $$/     $$  \$$ /    $$ |$$ /      |    $$ |      $$       |   $$  $$/
#    $$$$$$$$$/ $$$$$$$$$/ $$/    $$/ $$/        $$/$$$$$$$$$ / $$$$$$$$$/     $$/     $$$$$$$$$$/     $$$$/


import argparse
import numpy as np
import PySimpleGUI as sg

APP_WIDTH = 90
LIEN_WIDTH = 150
TEXT_FONT = "Helvetica"
TEXT_SIZE = 12
TEXT_BOX_SIZE = 10
TEXT_HEAD_SIZE = 14
HEADING_SIZE = 16

BLUE_BUTTON_COLOR = '#FFFFFF on #2196f2'
RED_BUTTON_COLOR = '#FFFFFF on #fa5f55'
GREEN_BUTTON_COLOR = '#FFFFFF on #00c851'
SEPARATOR_COLOR = '#FFFFFF'
LIGHT_GRAY_BUTTON_COLOR = f'#212021 on #e0e0e0'
DARK_GRAY_BUTTON_COLOR = '#e0e0e0 on #212021'


def make_gui(theme):
    sg.theme(theme)

    ################################################    Market layout    ###############################################
    market_layout = [[sg.Text('Market', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Dates:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('(2019, 1, 1) - (2021, 2, 15)',
                               key='-DATES-',
                               size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left',
                               font=(TEXT_FONT, TEXT_SIZE))],
                     [sg.HSeparator(color=SEPARATOR_COLOR)],]

    ################################################    Broker layout    ###############################################
    broker_layout = [[sg.Text('Broker', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Buy fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('0.08', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-BUY-'),
                      sg.Text('Sell fee:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('0.08', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-SELL-'),
                      sg.Text('Tax:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('25', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE),
                               key='-TAX-')],
                     [sg.HSeparator(color=SEPARATOR_COLOR)],]

    ################################################    Trader layout    ###############################################
    trader_layout = [[sg.Text('Trader', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Tickers:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('AAPL, GOOG, TSLA, ORCL', size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left', font=(TEXT_FONT, TEXT_SIZE), key='-TICKERS-')],
                     [sg.Text('Ratios:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('25, 25, 25, 25', size=APP_WIDTH-TEXT_BOX_SIZE,
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

    ################################################    Input layout    ################################################
    input_layout = [[sg.HSeparator(color=SEPARATOR_COLOR)]]
    input_layout += market_layout + broker_layout + trader_layout
    input_layout += [[sg.Text('Progress Bar', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE),
                              size=APP_WIDTH)],
                     [sg.ProgressBar(10000, orientation='h', size=(80, 20), bar_color=('green', 'white'),
                                     key='-PROGRESS BAR-')]]
    input_layout += [[sg.Button('RUN', size=(77, 2), button_color=GREEN_BUTTON_COLOR, k='-RUN-'),
                      sg.Button('HELP', size=(11, 2), k='-HELP-'),
                      sg.Button('EXIT', size=(12, 2), k='-EXIT-')]]

    ################################################    Logger layout    ###############################################
    logging_layout = [[sg.Text('Logger', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                      [sg.Output(size=(140, 50), font='Courier 8')]]

    ################################################    Plots layout    ################################################
    graphing_layout = [[sg.Text('Plots', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                       [sg.Canvas(size=(40, 10), key='-PLOTS-')]]

    ################################################    Theming layout    ##############################################
    theme_layout = [[sg.Text("See how elements look under different themes by choosing a different theme here!")],
                    [sg.Listbox(values=sg.theme_list(),
                                size=(20, 12),
                                key='-THEME LISTBOX-',
                                enable_events=True)],
                    [sg.Button("Set Theme")]]

    ################################################    Main layout    ###############################################
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
            print("[LOG] Clicked HELP!")
            #sg.popup('This should be ',
            #         'a help pop up !')

            dates = values['-DATES-'].split('-')
            start_date = np.array(dates[0].strip()[1:-1].split(','), dtype=np.int)
            end_date = np.array(dates[1].strip()[1:-1].split(','), dtype=np.int)
            buy_fee = np.float(values['-BUY-'])
            sell_fee = np.float(values['-SELL-'])
            tax = np.float(values['-TAX-'])
            tickers = [ticker.strip() for ticker in values['-TICKERS-'].split(',')]
            ratios = [np.float(ratio.strip()) for ratio in values['-RATIOS-'].split(',')]
            periods = [np.int(period.strip()) for period in values['-PERIODS-'].split(',')]
            sell_strategy = values['-SELL STRATEGY-']
            verbose = values['-VERBOSE-']
            print(start_date)
            print(end_date)
            print(buy_fee)
            print(sell_fee)
            print(tax)
            print(tickers)
            print(ratios)
            print(periods)
            print(sell_strategy)
            print(verbose)


        elif event == '-RUN-':
            # Running the simulation
            print("======  Starting BackTesting Simulation  ======")
            progress_bar = window['-PROGRESS BAR-']

            # extract execution arguments from variables
            variables = ['-DATES-', '-BUY-', '-SELL-', '-TAX-', '-TICKERS-', '-RATIOS-', '-PERIODS-', '-SELL STRATEGY-',
                         '-VERBOSE-', '-PLOTS-', '-TAB GROUP-']

            dates = values['-DATES-'].split('-')
            start_date = tuple(np.array(dates[0].strip()[1:-1].split(','), dtype=np.int))
            end_date = tuple(np.array(dates[1].strip()[1:-1].split(','), dtype=np.int))
            buy_fee = np.float(values['-BUY-'])
            sell_fee = np.float(values['-SELL-'])
            tax = np.float(values['-TAX-'])
            tickers = [ticker.strip() for ticker in values['-TICKERS-'].split(',')]
            ratios = [np.float(ratio.strip()) for ratio in values['-RATIOS-'].split(',')]
            periods = [np.int(period.strip()) for period in values['-PERIODS-'].split(',')]
            sell_strategy = values['-SELL STRATEGY-']
            verbose = values['-VERBOSE-']

            # verify input values are in bounds

            # arrange input variables to parser arguments





            for i in range(1, 10000, int(10000 / 1000)):
                print("[LOG] Updating progress bar by 1 step (" + str(i) + ")")
                progress_bar.UpdateBar(i + 1)
            print("[LOG] Progress bar complete!")
        elif event == "-PLOTS-":
            pass
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
