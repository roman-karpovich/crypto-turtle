import argparse
import sys
import time
import datetime
import json
import requests
import pandas as pd
import hmac
import hashlib
import matplotlib.pyplot as plt
from pybit.unified_trading import HTTP as TradingHTTP
from loguru import logger

# ------------------- CONFIGURATION -------------------
# Ensure config.json contains {"api_key": "...", "api_secret": "..."}
INTERVAL = 'D'  # Daily candles
BACKTEST_INTERVAL = '60'  # 1-hour candles for backtesting simulation
HIST_LIMIT = 200  # Number of historical bars to fetch (for live trading)
BACKTEST_LIMIT = 2000  # Number of days for backtesting (if available)
BACKTEST_POSITION_SIZE = 10  # Backtest position size in usd
RISK_PERCENT = 0.9  # Risk 2% of equity per trade
RECV_WINDOW = 5000  # Recv window in ms

# Fixed confirmation thresholds
RSI_LONG_THRESHOLD = 50
RSI_SHORT_THRESHOLD = 50
FIXED_ATR_PERCENT_THRESHOLD = 0.005  # 0.5% of price

# Partial exit ratio: fraction of position to exit on exit signal
PARTIAL_EXIT_RATIO = 0.5

# Pyramid parameters
PYRAMID_THRESHOLD = 0.03  # 3% favorable move required to add on
PYRAMID_MAX_COUNT = 3  # Maximum additional orders per position
PYRAMID_ORDER_FACTOR = 0.5  # Additional order is 50% of base risk-based order size

# Volume filter parameters (if desired)
VOL_PERIOD = 20  # Volume moving average period
VOL_MULTIPLIER = 1.2  # Current volume must exceed 1.2x its 20-day average

# Define thresholds for volatility (ATR% values) to map leverage
LOW_VOL_THRESHOLD = 0.01  # 1% ATR or lower -> highest leverage
HIGH_VOL_THRESHOLD = 0.05  # 5% ATR or higher -> lowest leverage

# Maintenance Margin (for liquidation checks in backtest)
MAINTENANCE_MARGIN = 0.05  # Default is 5%

# Symbols to trade (USDT perpetual futures)
SYMBOLS = [
    "XLMUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ETHUSDT"
]
# -----------------------------------------------------

# Global dictionaries for pyramid data and analytics
pyramid_data = {}
analytics = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_profit": 0.0
}

# Load API credentials from config.json
with open('config.json') as f:
    config = json.load(f)
API_KEY = config['api_key']
API_SECRET = config['api_secret']

# Initialize Bybit Unified Trading session
trading_session = TradingHTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)


# ------------------- RETRY WRAPPER -------------------
def retry_request(func, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning("Error in {}: {}. Retry {}/{}".format(func.__name__, e, i + 1, retries))
            time.sleep(delay * (2 ** i))
    raise Exception("Function {} failed after {} retries.".format(func.__name__, retries))


# ------------------- API FUNCTIONS -------------------
def sign_request(params, api_secret):
    sorted_params = sorted(params.items())
    param_str = '&'.join(f"{k}={v}" for k, v in sorted_params)
    return hmac.new(api_secret.encode('utf-8'),
                    param_str.encode('utf-8'),
                    hashlib.sha256).hexdigest()


def check_response(response):
    ret_code = response.get("ret_code", response.get("retCode", None))
    if ret_code is None:
        raise Exception("Response missing ret_code/retCode: " + json.dumps(response))
    if ret_code != 0:
        ret_msg = response.get("ret_msg", response.get("retMsg", ""))
        raise Exception(ret_msg)


def log_trade(action, details):
    logger.info("Action: {} - Details: {}", action, details)


def get_historical_klines(symbol, interval, limit=HIST_LIMIT):
    url = "https://api.bybit.com/v5/market/kline"
    max_rows_per_request = 1000  # Bybit's API limit per request
    all_data = []
    end_timestamp = None  # Used for pagination

    while len(all_data) < limit:
        # Determine how many rows are left to fetch
        rows_to_fetch = min(max_rows_per_request, limit - len(all_data))
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": rows_to_fetch
        }
        if end_timestamp:
            params["end"] = end_timestamp  # Pagination

        logger.debug("Requesting klines for {} with params: {}", symbol, params)
        response = retry_request(requests.get, url, params=params).json()
        check_response(response)

        data = response["result"]["list"]
        if not data:
            logger.warning("No more historical data found for {}.", symbol)
            break  # Exit if no more data is returned

        # Append only new unique records
        new_data = []
        for row in data:
            if len(all_data) == 0 or row[0] < all_data[-1][0]:  # Only add older records
                new_data.append(row)

        if not new_data:
            logger.warning("No new unique data fetched. Ending pagination for {}.", symbol)
            break

        all_data.extend(new_data)

        # Set `end_timestamp` to the earliest timestamp retrieved - 1ms to avoid duplication
        end_timestamp = int(new_data[-1][0]) - 1

        # If we received less than the max limit, there’s no more data available.
        if len(new_data) < max_rows_per_request:
            break

        time.sleep(0.5)  # Small delay to avoid API rate limits

    # Convert data into a DataFrame
    if all_data and isinstance(all_data[0], list):
        columns = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(all_data, columns=columns)
    else:
        df = pd.DataFrame(all_data)

    if df.empty:
        logger.error("Failed to fetch historical klines for {}.", symbol)
        return df

    # Convert timestamps and numerical values
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit='ms')
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset="open_time", keep="first", inplace=True)  # Ensure no duplicate rows
    df.reset_index(drop=True, inplace=True)

    logger.debug("Fetched {} klines for {} (chronological):\n{}", len(df), symbol, df.tail())

    return df


def compute_atr(df, period=14):
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(row['high'] - row['low'],
                                        abs(row['high'] - row['prev_close']),
                                        abs(row['low'] - row['prev_close'])), axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    df.drop(['prev_close', 'tr'], axis=1, inplace=True)
    return df


def compute_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calculate_turtle_signals(df):
    df['20d_high'] = df['high'].rolling(window=20).max()
    df['20d_low'] = df['low'].rolling(window=20).min()
    df['10d_high'] = df['high'].rolling(window=10).max()
    df['10d_low'] = df['low'].rolling(window=10).min()
    df['prev_20d_high'] = df['20d_high'].shift(1)
    df['prev_20d_low'] = df['20d_low'].shift(1)
    df['prev_10d_high'] = df['10d_high'].shift(1)
    df['prev_10d_low'] = df['10d_low'].shift(1)

    df['long_entry'] = df['close'] > df['prev_20d_high']
    df['long_exit'] = df['close'] < df['prev_10d_low']
    df['short_entry'] = df['close'] < df['prev_20d_low']
    df['short_exit'] = df['close'] > df['prev_10d_high']

    df = compute_atr(df.copy(), period=14)
    df = compute_rsi(df.copy(), period=14)

    df['vol_avg'] = df['volume'].rolling(window=VOL_PERIOD).mean()
    df['vol_confirm'] = df['volume'] > (VOL_MULTIPLIER * df['vol_avg'])

    return df


def get_wallet_equity():
    response = retry_request(trading_session.get_wallet_balance, coin="USDT", accountType="UNIFIED")
    check_response(response)
    result = response["result"]
    if "list" in result:
        for account in result["list"]:
            if "coin" in account and isinstance(account["coin"], list):
                for coin_info in account["coin"]:
                    if coin_info.get("coin") == "USDT":
                        return float(coin_info["equity"])
    raise Exception("USDT not found in wallet balance response.")


def get_current_position(symbol):
    url = "https://api.bybit.com/v5/position/list"
    timestamp = int(time.time() * 1000)
    params = {
        "api_key": API_KEY,
        "timestamp": timestamp,
        "recv_window": RECV_WINDOW,
        "symbol": symbol,
        "accountType": "UNIFIED",
        "category": "linear"
    }
    params["sign"] = sign_request(params, API_SECRET)
    logger.debug("Requesting positions for {} with params: {}", symbol, params)
    response = retry_request(requests.get, url, params=params).json()
    check_response(response)
    logger.debug("Position response for {}: {}", symbol, response)
    positions = response["result"].get("list", [])
    for pos in positions:
        if float(pos["size"]) > 0:
            logger.info("Found open position for {}: {}", symbol, pos)
            return pos
    return None


def get_latest_price(symbol):
    df = get_historical_klines(symbol, INTERVAL, limit=1)
    latest_price = float(df.iloc[-1]['close'])
    logger.debug("Latest price for {}: {}", symbol, latest_price)
    return latest_price


# ------------------- LEVERAGE CALCULATION -------------------
def calculate_leverage(atr_percent):
    """
    Returns a leverage factor between 1x and 20x, depending on ATR%.
    Lower ATR% => higher leverage; higher ATR% => lower leverage.
    """
    min_leverage = 1
    max_leverage = 20

    if atr_percent <= LOW_VOL_THRESHOLD:
        return max_leverage  # 20x
    elif atr_percent >= HIGH_VOL_THRESHOLD:
        return min_leverage  # 1x
    else:
        # Linear interpolation between min_leverage and max_leverage
        ratio = (atr_percent - LOW_VOL_THRESHOLD) / (HIGH_VOL_THRESHOLD - LOW_VOL_THRESHOLD)
        # ratio goes from 0.0 to 1.0 as atr_percent goes from LOW_VOL_THRESHOLD to HIGH_VOL_THRESHOLD
        return max_leverage - (ratio * (max_leverage - min_leverage))


# ------------------- POSITION SIZE CALCULATION -------------------
def calculate_position_size(entry_price, stop_loss_price, equity, leverage):
    """
    Calculate position size by scaling the risk amount by the leverage factor.
    Note: Using leverage here increases the notional position while still risking a fixed percent of equity.
    """
    risk_amount = equity * RISK_PERCENT
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0
    # Scale risk_amount by leverage to compute a larger notional exposure.
    pos_size = (risk_amount * leverage) / risk_per_unit
    logger.debug("Calculated position size with leverage {}x: {} (Scaled Risk: {}, Per Unit Risk: {})",
                 leverage, pos_size, risk_amount * leverage, risk_per_unit)
    return round(pos_size, 8)


def place_order(symbol, side, order_type, qty, price=None, reduce_only=False):
    try:
        logger.debug("Placing order on {}: side={}, order_type={}, qty={}, price={}, reduce_only={}",
                     symbol, side, order_type, qty, price, reduce_only)
        # Use the unified trading API's order creation method.
        response = retry_request(trading_session.create_order,
                                 symbol=symbol,
                                 side=side,
                                 order_type=order_type,
                                 qty=qty,
                                 price=price,
                                 time_in_force="GoodTillCancel",
                                 reduce_only=reduce_only)
        check_response(response)
        log_trade("place_order", response)
        logger.info("Order placed on {}: {}", symbol, response)
        return response
    except Exception as e:
        logger.error("Error placing order on {}: {}", symbol, e)
        log_trade("order_error", str(e))
        return None


def set_trading_stop(symbol, stop_loss=None, trailing_stop=None):
    try:
        logger.debug("Setting trading stop on {}: SL={}, trailing_stop={}", symbol, stop_loss, trailing_stop)
        response = retry_request(trading_session.set_trading_stop,
                                 symbol=symbol,
                                 stop_loss=stop_loss,
                                 trailing_stop=trailing_stop)
        check_response(response)
        log_trade("set_trading_stop", response)
        logger.info("Trading stop set on {}: {}", symbol, response)
        return response
    except Exception as e:
        logger.error("Error setting trading stop on {}: {}", symbol, e)
        log_trade("trading_stop_error", str(e))
        return None


def update_trailing_stop(symbol, new_stop_loss):
    try:
        logger.debug("Updating trailing stop on {}: new SL={}", symbol, new_stop_loss)
        response = retry_request(trading_session.set_trading_stop,
                                 symbol=symbol,
                                 stop_loss=new_stop_loss)
        check_response(response)
        log_trade("update_stop_loss", response)
        logger.info("Stop loss updated on {} to {}", symbol, new_stop_loss)
        return response
    except Exception as e:
        logger.error("Error updating stop loss on {}: {}", symbol, e)
        log_trade("update_stop_loss_error", str(e))
        return None


def close_position(symbol, qty):
    logger.debug("Closing position on {}: qty={}", symbol, qty)
    return place_order(symbol, side="Sell", order_type="Market", qty=qty, reduce_only=True)


# ------------------- ANALYTICS FUNCTIONS -------------------
def record_trade(symbol, direction, entry_price, exit_price, qty):
    profit = (exit_price - entry_price) * qty if direction == "long" else (entry_price - exit_price) * qty
    analytics["total_trades"] += 1
    if profit > 0:
        analytics["winning_trades"] += 1
    else:
        analytics["losing_trades"] += 1
    analytics["total_profit"] += profit
    logger.info("[Analytics] Trade recorded for {}: {} trade, Entry: {}, Exit: {}, Qty: {}, Profit: {:.2f}",
                symbol, direction, entry_price, exit_price, qty, profit)


def log_analytics():
    if analytics["total_trades"] > 0:
        win_rate = analytics["winning_trades"] / analytics["total_trades"] * 100
        avg_profit = analytics["total_profit"] / analytics["total_trades"]
    else:
        win_rate = 0
        avg_profit = 0
    logger.info(
        "[Analytics] Total Trades: {}, Wins: {}, Losses: {}, Win Rate: {:.2f}%, Total Profit: {:.2f}, Average Profit: {:.2f}",
        analytics["total_trades"], analytics["winning_trades"], analytics["losing_trades"], win_rate,
        analytics["total_profit"], avg_profit)


# ------------------- PLOTTING FUNCTION -------------------
def plot_signals(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['close'], label='Close Price', color='black', linewidth=1)
    plt.plot(df['open_time'], df['prev_20d_high'], label='Prev 20d High', color='green', linestyle='--')
    plt.plot(df['open_time'], df['prev_20d_low'], label='Prev 20d Low', color='red', linestyle='--')
    plt.plot(df['open_time'], df['prev_10d_high'], label='Prev 10d High', color='orange', linestyle=':')
    plt.plot(df['open_time'], df['prev_10d_low'], label='Prev 10d Low', color='blue', linestyle=':')

    # Draw exit signals first
    plt.scatter(df[df['long_exit']]['open_time'], df[df['long_exit']]['close'],
                marker='v', color='red', s=100, label='Long Exit')
    plt.scatter(df[df['short_exit']]['open_time'], df[df['short_exit']]['close'],
                marker='^', color='orange', s=100, label='Short Exit')

    # Draw entry signals on top; add slight offset for short entry markers if needed
    plt.scatter(df[df['long_entry']]['open_time'], df[df['long_entry']]['close'],
                marker='^', color='green', s=100, label='Long Entry')
    short_entry_offset = df[df['short_entry']]['close'] - 0.002 * df[df['short_entry']]['close']
    plt.scatter(df[df['short_entry']]['open_time'], short_entry_offset,
                marker='v', color='blue', s=100, label='Short Entry')

    plt.title(f"Historical Signals for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    filename = f"plot_signals/{symbol}.png"
    plt.savefig(filename)
    plt.close()
    logger.info("Plot saved to {}", filename)


# ------------------- BACKTESTING FUNCTION -------------------
# ------------------- PLOTTING FUNCTION FOR BACKTEST -------------------
def plot_backtest_results(symbol, df, trades):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['close'], label='Close Price', color='black', linewidth=1)

    # Plot each trade's entry and exit, and annotate the exit point with profit
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        profit = trade['profit']

        if trade['direction'] == 'long':
            plt.scatter(entry_time, entry_price, marker='^', color='green', s=100, label='Long Entry')
            plt.scatter(exit_time, exit_price, marker='v', color='red', s=100, label='Long Exit')
            plt.annotate(f"Profit: {profit:.2f}", xy=(exit_time, exit_price),
                         xytext=(exit_time, exit_price * 0.98),
                         arrowprops=dict(arrowstyle="->", color='red'),
                         fontsize=8, color='red')
        elif trade['direction'] == 'short':
            plt.scatter(entry_time, entry_price, marker='v', color='red', s=100, label='Short Entry')
            plt.scatter(exit_time, exit_price, marker='^', color='green', s=100, label='Short Exit')
            plt.annotate(f"Profit: {profit:.2f}", xy=(exit_time, exit_price),
                         xytext=(exit_time, exit_price * 1.02),
                         arrowprops=dict(arrowstyle="->", color='green'),
                         fontsize=8, color='green')

    # Remove duplicate legend entries.
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.title(f"Backtest Results for {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    filename = f"backtest_results/{symbol}.png"
    plt.savefig(filename)
    plt.close()
    logger.info("Backtest plot saved as {}", filename)


def compute_liquidation_price_long(entry_price, leverage):
    # Simplified: if price drops enough that we lose (1 - MAINTENANCE_MARGIN) of our margin
    # margin fraction = 1 / leverage. So if price moves down by that fraction (minus maintenance margin), liquidate.
    # price < entry_price * (1 - (1/leverage)*(1 - MAINTENANCE_MARGIN)) => liquidation
    return entry_price * (1 - (1 / leverage) * (1 - MAINTENANCE_MARGIN))


def compute_liquidation_price_short(entry_price, leverage):
    # For short, if price rises enough that we lose (1 - MAINTENANCE_MARGIN) of margin => liquidation
    # price > entry_price + (entry_price*(1/leverage)*(1 - MAINTENANCE_MARGIN)) => liquidation
    return entry_price * (1 + (1 / leverage) * (1 - MAINTENANCE_MARGIN))


# ------------------- BACKTESTING FUNCTION (Improved) -------------------
def backtest_symbol(symbol):
    logger.info("Starting backtest for {}", symbol)

    df_daily = get_historical_klines(symbol, INTERVAL, limit=BACKTEST_LIMIT)
    df_daily = calculate_turtle_signals(df_daily)
    df_daily.sort_values("open_time", inplace=True)

    df_5m = get_historical_klines(symbol, BACKTEST_INTERVAL, limit=BACKTEST_LIMIT * 24)
    df_5m.sort_values("open_time", inplace=True)

    daily_signal_cols = ["open_time", "prev_20d_high", "prev_10d_low", "prev_20d_low", "prev_10d_high", "rsi", "atr",
                         "long_entry", "short_entry", "long_exit", "short_exit"]
    df_merged = pd.merge_asof(df_5m, df_daily[daily_signal_cols], on="open_time", direction="backward")

    trades = []
    position = None

    for idx, row in df_merged.iterrows():
        # Candle info
        candle_open = row['open']
        candle_high = row['high']
        candle_low = row['low']
        candle_close = row['close']
        candle_time = row['open_time']
        # worst-case average mid price
        candle_mid = (candle_high + candle_low) / 2 if (candle_high and candle_low) else candle_close

        # daily signals
        breakout_long = row['prev_20d_high']
        exit_long_thresh = row['prev_10d_low']
        breakout_short = row['prev_20d_low']
        exit_short_thresh = row['prev_10d_high']
        rsi = row['rsi']
        atr = row['atr']

        if pd.isna(atr) or candle_open == 0:
            # skip incomplete data
            continue

        atr_percent = atr / candle_close if candle_close != 0 else 0.0
        leverage = calculate_leverage(atr_percent)

        # Define the path for worst-case price movement in this candle
        if position is None:
            # Evaluate entry signals
            # 1) Potential LONG
            if (
                    row['long_entry'] and
                    rsi > RSI_LONG_THRESHOLD and
                    (atr_percent > FIXED_ATR_PERCENT_THRESHOLD)
            ):
                # We see if candle hits breakout_long in a worst-case sequence: open -> low -> mid -> high -> close
                # If the breakout occurs at or below the high, we can assume an entry at breakout_long
                # We'll do a simplified check if candle_open >= breakout_long or if candle_high >= breakout_long
                if candle_open >= breakout_long:
                    entry_price = candle_open
                elif candle_high >= breakout_long:
                    entry_price = breakout_long
                else:
                    entry_price = None

                if entry_price is not None:
                    stop_loss_price = exit_long_thresh
                    qty = calculate_position_size(entry_price, stop_loss_price, BACKTEST_POSITION_SIZE, leverage)
                    if qty > 0:
                        position = {
                            'direction': 'long',
                            'entry_price': entry_price,
                            'entry_time': candle_time,
                            'qty': qty,
                            'stop_loss': stop_loss_price,
                            'liquidation_price': compute_liquidation_price_long(entry_price, leverage)
                        }
            # 2) Potential SHORT
            if (position is None and
                    row['short_entry'] and
                    rsi < RSI_SHORT_THRESHOLD and
                    (atr_percent > FIXED_ATR_PERCENT_THRESHOLD)
            ):
                if candle_open <= breakout_short:
                    entry_price = candle_open
                elif candle_low <= breakout_short:
                    entry_price = breakout_short
                else:
                    entry_price = None

                if entry_price is not None:
                    stop_loss_price = exit_short_thresh
                    qty = calculate_position_size(entry_price, stop_loss_price, BACKTEST_POSITION_SIZE, leverage)
                    if qty > 0:
                        position = {
                            'direction': 'short',
                            'entry_price': entry_price,
                            'entry_time': candle_time,
                            'qty': qty,
                            'stop_loss': stop_loss_price,
                            'liquidation_price': compute_liquidation_price_short(entry_price, leverage)
                        }
        else:
            # We have an open position. We'll simulate the worst-case path.
            path = []
            if position['direction'] == 'long':
                path = [candle_open, candle_low, candle_mid, candle_high, candle_close]
            else:  # short
                path = [candle_open, candle_high, candle_mid, candle_low, candle_close]

            exit_triggered = False

            for step_price in path:
                # Check liquidation first
                if position['direction'] == 'long':
                    if step_price <= position['liquidation_price']:
                        # Liquidation
                        trade = {
                            'direction': 'long',
                            'entry_price': position['entry_price'],
                            'exit_price': position['liquidation_price'],
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (position['liquidation_price'] - position['entry_price']) * position['qty']
                        }
                        trades.append(trade)
                        position = None
                        exit_triggered = True
                        break
                    # Check stop
                    if step_price <= position['stop_loss']:
                        trade = {
                            'direction': 'long',
                            'entry_price': position['entry_price'],
                            'exit_price': position['stop_loss'],
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (position['stop_loss'] - position['entry_price']) * position['qty']
                        }
                        trades.append(trade)
                        position = None
                        exit_triggered = True
                        break
                else:  # short
                    if step_price >= position['liquidation_price']:
                        # Liquidation
                        trade = {
                            'direction': 'short',
                            'entry_price': position['entry_price'],
                            'exit_price': position['liquidation_price'],
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (position['entry_price'] - position['liquidation_price']) * position['qty']
                        }
                        trades.append(trade)
                        position = None
                        exit_triggered = True
                        break
                    # Check stop
                    if step_price >= position['stop_loss']:
                        trade = {
                            'direction': 'short',
                            'entry_price': position['entry_price'],
                            'exit_price': position['stop_loss'],
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (position['entry_price'] - position['stop_loss']) * position['qty']
                        }
                        trades.append(trade)
                        position = None
                        exit_triggered = True
                        break

            # If still in position after path, check if daily signal says exit
            if not exit_triggered and position is not None:
                if position['direction'] == 'long':
                    # daily exit signal
                    if row['long_exit']:
                        # close at candle_close
                        trade = {
                            'direction': 'long',
                            'entry_price': position['entry_price'],
                            'exit_price': candle_close,
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (candle_close - position['entry_price']) * position['qty']
                        }
                        trades.append(trade)
                        position = None
                else:
                    if row['short_exit']:
                        trade = {
                            'direction': 'short',
                            'entry_price': position['entry_price'],
                            'exit_price': candle_close,
                            'entry_time': position['entry_time'],
                            'exit_time': candle_time,
                            'profit': (position['entry_price'] - candle_close) * position['qty']
                        }
                        trades.append(trade)
                        position = None

    # End of data loop, close any open position at last price? (Optional) - We'll skip unless we want forced exit.

    total_trades = len(trades)
    total_profit = sum(tr['profit'] for tr in trades)
    wins = [t for t in trades if t['profit'] > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_profit = (total_profit / total_trades) if total_trades > 0 else 0

    logger.info(
        "[Backtest] {}: Total Trades: {}, Win Rate: {:.2f}%, Total Profit: {:.2f}, Average Profit: {:.2f}",
        symbol, total_trades, win_rate, total_profit, avg_profit
    )

    plot_backtest_results(symbol, df_merged, trades)
    return trades


# ------------------- LIVE TRADING FUNCTION -------------------
def trade_symbol(symbol):
    # Fetch daily data and calculate signals
    df = get_historical_klines(symbol, INTERVAL, limit=HIST_LIMIT)
    df = calculate_turtle_signals(df)
    df.sort_values("open_time", inplace=True)

    # Determine whether to use yesterday's confirmed daily candle for signals
    now_date = datetime.datetime.now(datetime.timezone.utc).date()
    if df.iloc[-1]['open_time'].date() == now_date:
        confirmed_candle = df.iloc[-2]
        logger.info("[{}] Using yesterday's confirmed daily candle for signals.", symbol)
        using_yesterday = True
    else:
        confirmed_candle = df.iloc[-1]
        logger.info("[{}] Using latest daily candle for signals.", symbol)
        using_yesterday = False

    current_price = get_latest_price(symbol)
    equity = get_wallet_equity()
    # Use confirmed candle's ATR and close for volatility calculations
    atr_percent = (confirmed_candle['atr'] / confirmed_candle['close']) if confirmed_candle['close'] != 0 else 0.0

    # Adaptive ATR threshold remains as before.
    recent_atr_percent = df.iloc[-14:]['atr'].mean() / df.iloc[-14:]['close'].mean()
    adaptive_atr_threshold = max(FIXED_ATR_PERCENT_THRESHOLD, recent_atr_percent)

    # Calculate dynamic leverage based on ATR% volatility.
    leverage = calculate_leverage(atr_percent)

    logger.info("[{}] {} | Price: {} | Equity: {} | ATR%: {:.2%} (Adaptive: {:.2%}) | RSI: {:.2f} | Leverage: {}x",
                symbol,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                current_price, equity, atr_percent, adaptive_atr_threshold,
                confirmed_candle['rsi'], round(leverage, 2))

    # If we're using yesterday's confirmed candle, ensure today's unclosed candle has reached the breakout.
    if using_yesterday:
        # For a long signal, require that today's price has reached yesterday's breakout level.
        if confirmed_candle['long_entry']:
            breakout_level = confirmed_candle['prev_20d_high']
            if current_price < breakout_level:
                logger.info("[{}] Today's price ({}) has not broken out above {} for long entry. Waiting...", symbol,
                            current_price, breakout_level)
                return  # Skip entry
        # For a short signal, require that today's price has reached below yesterday's breakout level.
        if confirmed_candle['short_entry']:
            breakout_level = confirmed_candle['prev_20d_low']
            if current_price > breakout_level:
                logger.info("[{}] Today's price ({}) has not broken below {} for short entry. Waiting...", symbol,
                            current_price, breakout_level)
                return  # Skip entry

    position = get_current_position(symbol)

    if position is None:
        if symbol in pyramid_data:
            del pyramid_data[symbol]
        # Evaluate long entry using confirmed daily candle signals.
        if (confirmed_candle['long_entry'] and confirmed_candle[
            'rsi'] > RSI_LONG_THRESHOLD and atr_percent > adaptive_atr_threshold):
            logger.info("[{}] Confirmed LONG entry signal from closed candle.", symbol)
            stop_loss_price = confirmed_candle['prev_10d_low']
            entry_price = current_price
            pos_size = calculate_position_size(entry_price, stop_loss_price, equity, leverage)
            if pos_size <= 0:
                logger.warning("[{}] Long pos_size <= 0, skipping entry.", symbol)
            else:
                logger.info("[{}] LONG size: {} (Entry: {}, Stop: {}, Leverage: {}x)", symbol, pos_size, entry_price,
                            stop_loss_price, round(leverage, 2))
                order_resp = place_order(symbol, side="Buy", order_type="Market", qty=pos_size)
                if order_resp:
                    set_trading_stop(symbol, stop_loss=round(stop_loss_price, 4))
                    pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        # Evaluate short entry using confirmed candle signals.
        elif (confirmed_candle['short_entry'] and confirmed_candle[
            'rsi'] < RSI_SHORT_THRESHOLD and atr_percent > adaptive_atr_threshold):
            logger.info("[{}] Confirmed SHORT entry signal from closed candle.", symbol)
            stop_loss_price = confirmed_candle['prev_10d_high']
            entry_price = current_price
            pos_size = calculate_position_size(entry_price, stop_loss_price, equity, leverage)
            if pos_size <= 0:
                logger.warning("[{}] Short pos_size <= 0, skipping entry.", symbol)
            else:
                logger.info("[{}] SHORT size: {} (Entry: {}, Stop: {}, Leverage: {}x)", symbol, pos_size, entry_price,
                            stop_loss_price, round(leverage, 2))
                order_resp = place_order(symbol, side="Sell", order_type="Market", qty=pos_size)
                if order_resp:
                    set_trading_stop(symbol, stop_loss=round(stop_loss_price, 4))
                    pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        else:
            logger.info("[{}] No confirmed entry signal. Waiting...", symbol)
    else:
        # Existing position management remains unchanged.
        pos_side = position.get("side", "")
        entry_price = float(position["entry_price"])
        pos_qty = float(position["size"])
        if symbol not in pyramid_data:
            pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        if pos_side == "Buy":
            if confirmed_candle['long_exit']:
                logger.info("[{}] Long exit signal triggered. Executing partial exit.", symbol)
                exit_qty = pos_qty * PARTIAL_EXIT_RATIO
                close_position(symbol, qty=exit_qty)
                record_trade(symbol, "long", entry_price, current_price, exit_qty)
            elif current_price > pyramid_data[symbol]['baseline'] * (1 + PYRAMID_THRESHOLD) and pyramid_data[symbol][
                'count'] < PYRAMID_MAX_COUNT:
                additional_size = calculate_position_size(current_price, confirmed_candle['prev_10d_low'], equity,
                                                          leverage) * PYRAMID_ORDER_FACTOR
                if additional_size > 0:
                    logger.info("[{}] Pyramid condition met for LONG. Adding additional order of size: {}", symbol,
                                additional_size)
                    order_resp = place_order(symbol, side="Buy", order_type="Market", qty=additional_size)
                    if order_resp:
                        pyramid_data[symbol]['baseline'] = current_price
                        pyramid_data[symbol]['count'] += 1
            elif current_price > entry_price * 1.05:
                new_sl = round(current_price * 0.98, 4)
                update_trailing_stop(symbol, new_stop_loss=new_sl)
            else:
                logger.info("[{}] Long position: no exit or trailing update triggered.", symbol)
        elif pos_side == "Sell":
            if confirmed_candle['short_exit']:
                logger.info("[{}] Short exit signal triggered. Executing partial exit.", symbol)
                exit_qty = pos_qty * PARTIAL_EXIT_RATIO
                close_position(symbol, qty=exit_qty)
                record_trade(symbol, "short", entry_price, current_price, exit_qty)
            elif current_price < pyramid_data[symbol]['baseline'] * (1 - PYRAMID_THRESHOLD) and pyramid_data[symbol][
                'count'] < PYRAMID_MAX_COUNT:
                additional_size = calculate_position_size(current_price, confirmed_candle['prev_10d_high'], equity,
                                                          leverage) * PYRAMID_ORDER_FACTOR
                if additional_size > 0:
                    logger.info("[{}] Pyramid condition met for SHORT. Adding additional order of size: {}", symbol,
                                additional_size)
                    order_resp = place_order(symbol, side="Sell", order_type="Market", qty=additional_size)
                    if order_resp:
                        pyramid_data[symbol]['baseline'] = current_price
                        pyramid_data[symbol]['count'] += 1
            elif current_price < entry_price * 0.95:
                new_sl = round(current_price * 1.02, 4)
                update_trailing_stop(symbol, new_stop_loss=new_sl)
            else:
                logger.info("[{}] Short position: no exit or trailing update triggered.", symbol)
        else:
            logger.warning("[{}] Unknown position side: {}", symbol, pos_side)
    log_analytics()


def turtle_trading_bot():
    while True:
        try:
            logger.info("=== Starting new cycle for all symbols ===")
            for sym in SYMBOLS:
                trade_symbol(sym)
                time.sleep(2)
        except Exception as e:
            logger.error("Error in multi-symbol loop: {}", e)
            log_trade("bot_error", str(e))
        time.sleep(30 * 60)


def backtest():
    # Run backtesting for each symbol
    for sym in SYMBOLS:
        backtest_symbol(sym)
        time.sleep(2)
    log_analytics()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Trading Strategy Script with Verbose Logging Option")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug/verbose logging")
    parser.add_argument("-b", "--backtest", action="store_true", help="Enable backtest")
    args = parser.parse_args()

    # Configure logging level based on the verbose flag.
    # Remove the default Loguru handler and configure output to stderr.
    logger.remove()
    logger.add("logs/bot_debug.log", level="DEBUG", rotation="10 MB")
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Verbose logging enabled.")
    else:
        logger.add(sys.stderr, level="INFO")
        logger.info("Standard logging enabled.")

    # If run with argument "backtest", execute backtesting mode.
    if args.backtest:
        backtest()
    else:
        turtle_trading_bot()
