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
GREEN_BUTTON_COLOR ='#FFFFFF on #00c851'
LIGHT_GRAY_BUTTON_COLOR = f'#212021 on #e0e0e0'
DARK_GRAY_BUTTON_COLOR = '#e0e0e0 on #212021'


def make_gui(theme):
    sg.theme(theme)

    # tabs layouts
    market_layout = [[sg.Text('Market', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                     [sg.Text('Dates:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Input('(2019, 1, 1) - (2021, 2, 15)',
                               key='-DATES-',
                               size=APP_WIDTH-TEXT_BOX_SIZE,
                               justification='left',
                               font=(TEXT_FONT, TEXT_SIZE))],
                     [sg.Text('-'*LIEN_WIDTH, justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE))]]

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
                     [sg.Text('-'*LIEN_WIDTH, justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE))]]

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
                      sg.DropDown(list(['FIFO', 'LIFO', 'TAX_OPT']), size=(15, 10), enable_events=False,
                                  font=(TEXT_FONT, TEXT_SIZE), key='-LIST-')],
                     [sg.Text('Verbose:', size=TEXT_BOX_SIZE, justification='left', font=(TEXT_FONT, TEXT_SIZE)),
                      sg.Checkbox('', default=False, key='-VERBOSE-')],
                     [sg.Text('-'*LIEN_WIDTH, justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE))]]

    input_layout = [[sg.Text('-'*LIEN_WIDTH, justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE))]]
    input_layout += market_layout + broker_layout + trader_layout
    input_layout += [[sg.Text('Progress Bar', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
        [sg.ProgressBar(10000, orientation='h', size=(80, 20), bar_color=('green', 'white'), key='-PROGRESS BAR-')]]
    input_layout += [[sg.Button('GO', size=(35, 2), button_color=GREEN_BUTTON_COLOR),
                      sg.Button('STOP', size=(35, 2), button_color=RED_BUTTON_COLOR),
                      sg.Button('HELP', size=(11, 2)),
                      sg.Button('EXIT', size=(12, 2))]]

    logging_layout = [[sg.Text('Logger', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                      [sg.Output(size=(140, 50), font='Courier 8')]]

    graphing_layout = [[sg.Text('Plots', justification='center', font=(TEXT_FONT, TEXT_HEAD_SIZE), size=APP_WIDTH)],
                       [sg.Canvas(size=(40, 10), key='-CANVAS-')]]

    theme_layout = [[sg.Text("See how elements look under different themes by choosing a different theme here!")],
                    [sg.Listbox(values=sg.theme_list(),
                                size=(20, 12),
                                key='-THEME LISTBOX-',
                                enable_events=True)],
                    [sg.Button("Set Theme")]]

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
        event, values = window.read(timeout=100)

        if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
            print('============ Event = ', event, ' ==============')
            print('-------- Values Dictionary (key=value) --------')
            for key in values:
                print(key, ' = ', values[key])
        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break
        elif event == 'About':
            print("[LOG] Clicked About!")
            sg.popup('PySimpleGUI Demo All Elements',
                     'Right click anywhere to see right click menu',
                     'Visit each of the tabs to see available elements',
                     'Output of event and values can be see in Output tab',
                     'The event and values dictionary is printed after every event')
        elif event == 'Popup':
            print("[LOG] Clicked Popup Button!")
            sg.popup("You pressed a button!")
            print("[LOG] Dismissing Popup!")
        elif event == 'Test Progress bar':
            print("[LOG] Clicked Test Progress Bar!")
            progress_bar = window['-PROGRESS BAR-']
            for i in range(1000):
                print("[LOG] Updating progress bar by 1 step (" + str(i) + ")")
                progress_bar.UpdateBar(i + 1)
            print("[LOG] Progress bar complete!")
        elif event == "-GRAPH-":
            graph = window['-GRAPH-']  # type: sg.Graph
            graph.draw_circle(values['-GRAPH-'], fill_color='yellow', radius=20)
            print("[LOG] Circle drawn at: " + str(values['-GRAPH-']))
        elif event == "Open Folder":
            print("[LOG] Clicked Open Folder!")
            folder_or_file = sg.popup_get_folder('Choose your folder')
            sg.popup("You chose: " + str(folder_or_file))
            print("[LOG] User chose folder: " + str(folder_or_file))
        elif event == "Open File":
            print("[LOG] Clicked Open File!")
            folder_or_file = sg.popup_get_file('Choose your file')
            sg.popup("You chose: " + str(folder_or_file))
            print("[LOG] User chose file: " + str(folder_or_file))
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
