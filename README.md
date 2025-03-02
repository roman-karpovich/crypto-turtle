# Turtle Trading Bot for Bybit Futures

This Python trading bot implements a Turtle (Donchian) breakout strategy for USDT-margined perpetual futures on Bybit. The script calculates Donchian channels along with ATR and RSI indicators to confirm entry signals for both long and short positions. It logs detailed debug information, saves historical signal data to CSV files, and generates plots for manual review.

## Features

- **Donchian Channel Signals**
  - **Long Entry:** Close > Previous 20-day high  
  - **Long Exit:** Close < Previous 10-day low  
  - **Short Entry:** Close < Previous 20-day low  
  - **Short Exit:** Close > Previous 10-day high

- **Confirmation Filters**
  - Uses ATR (14-period) and RSI (14-period) for signal confirmation.
  - **Long Entry** requires RSI > 50 and ATR as a percentage of price > 0.5%.
  - **Short Entry** requires RSI < 50 and ATR as a percentage of price > 0.5%.

- **Futures Trading**
  - Supports opening and closing both long and short positions on Bybit USDT perpetual futures.

- **Multi-Symbol Support**
  - Processes multiple symbols (e.g., XLMUSDT, BTCUSDT, SOLUSDT) sequentially in one loop.

- **Extensive Logging & Debugging**
  - Uses Loguru to output detailed logs to both the console and a file (`bot_debug.log`).
  - Saves historical signal data to CSV files (e.g., `signals_XLMUSDT.csv`).
  - Generates plots (e.g., `plot_signals_XLMUSDT.png`) overlaying price charts with signal markers for visual debugging.
  - Exit signals are drawn first, then entry signals are overlaid to ensure clarity.

## Prerequisites

- Python 3.8 or later
- [pybit](https://github.com/bybit-exchange/pybit) (v5.9.0 or later recommended)
- [pandas](https://pandas.pydata.org/)
- [requests](https://docs.python-requests.org/)
- [loguru](https://github.com/Delgan/loguru)
- [matplotlib](https://matplotlib.org/)

## Installation

1. **Clone the Repository** (or download the script file):

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a file named `config.json` in the same directory as the script.
2. Add your Bybit API credentials in JSON format:

   ```json
   {
       "api_key": "YOUR_API_KEY_HERE",
       "api_secret": "YOUR_API_SECRET_HERE"
   }
   ```

3. Adjust any parameters in the script if needed:
   - **RSI thresholds** (default: 50 for both long and short)
   - **ATR percentage threshold** (default: 0.5% of price)
   - **Risk percentage per trade** (default: 2%)
   - **Symbols** in the `SYMBOLS` list

## Usage

Run the script with:

   ```bash
   python bybit_trading_bot.py
   ```

The bot will process each symbol in the `SYMBOLS` list sequentially. It will:

- Output detailed logs to the console and `bot_debug.log`.
- Save historical signal data to CSV files (e.g., `signals_XLMUSDT.csv`).
- Generate signal plots as PNG files (e.g., `plot_signals_XLMUSDT.png`) for visual review.

## Debugging & Manual Review

- **Log Output:**  
  Check `bot_debug.log` for detailed debug information including API responses, calculated indicators (ATR, RSI), and executed trades.

- **CSV Files:**  
  The bot saves historical data with computed signals to CSV files each cycle. Open these files in Excel or your preferred tool to review the signal history.

- **Plot Files:**  
  The bot generates plots that overlay the price chart with Donchian channel levels and markers for exit and entry signals. Exit signals are drawn first, then entry signals are overlaid on top, ensuring entry markers remain visible. Use these plots to visually verify if the signals match your expectations.

## Futures Trading Mechanics

In USDT-margined futures on Bybit:
- A **"Buy"** order opens or increases a **long** position.
- A **"Sell"** order opens or increases a **short** position.
- Exiting a long position is done with a **Sell** order with `reduce_only=True`, and exiting a short position with a **Buy** order with `reduce_only=True`.

## Disclaimer

**This bot is provided "as is" without warranty of any kind.**  
Before using it in a live trading environment, thoroughly test it on a demo or testnet account. Futures trading involves substantial risk and may not be suitable for all investors.
