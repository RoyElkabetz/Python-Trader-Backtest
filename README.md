# Python Trader Backtest

Python Trader Backtest is an app for backtesting simple trading strategies based on historical trading data from the yfinance python package. The application enables variations of portfolio periodic balancing with a weighted averaged portfolio value for a variety of selling strategies like, FIFO (First In First Out), LIFO (First In Lirst Out) or TAX_OPT that picks which stock to sell next by minimizing the amount of tax payed in the transaction.



## Notebook

| #   | file            | Subject                                         | Colab             | Nbviewer               |
|:----:|:--------------:|:------------------------------------------------:|:-----------------:|:---------------------:|
| 1   | `backtesting_simulator.ipynb` | Runing the full simulator in Jupyter notebook   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)]()        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)]()|


## Command Line API


## The App
This repository also contains a simple (and a bit ugly) python app with the same functionality of the original API. The app got three windows:
- Input window which contain the input parameters to the market, broker and trader classes
<img src="assets/gui_in_progress.png" width="1000" height="">

- Output window for textual progress display 
<img src="assets/gui_simulation_output.png" width="1000" height="">

- Plots window for ploting the simulation results
<img src="assets/gui_plot_results.png" width="1000" height="">






 
