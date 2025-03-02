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
INTERVAL = 'D'              # Daily candles
HIST_LIMIT = 200            # Number of historical bars to fetch
RISK_PERCENT = 0.02         # Risk 2% of equity per trade
RECV_WINDOW = 5000          # Recv window in ms

# Fixed confirmation thresholds
RSI_LONG_THRESHOLD = 50
RSI_SHORT_THRESHOLD = 50
FIXED_ATR_PERCENT_THRESHOLD = 0.005  # 0.5% of price

# Partial exit ratio (percentage of position to exit on exit signal)
PARTIAL_EXIT_RATIO = 0.5

# Pyramid parameters (as in previous versions)
PYRAMID_THRESHOLD = 0.03      # 3% favorable move to add on
PYRAMID_MAX_COUNT = 3         # Maximum pyramid additions per position
PYRAMID_ORDER_FACTOR = 0.5    # Additional order is 50% of base risk-based order size

# Volume filter parameters (if used)
VOL_PERIOD = 20                 
VOL_MULTIPLIER = 1.2            

# Symbols to trade (USDT perpetual futures)
SYMBOLS = [
    "XLMUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "ETHUSDT"
]
# -----------------------------------------------------

logger.add("bot_debug.log", level="DEBUG", rotation="1 MB")

# Global dictionary to track pyramid data per symbol
pyramid_data = {}

# Global analytics dictionary to track trade performance
analytics = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_profit": 0.0  # profit in USDT (or relative units)
}

# ------------------- RETRY WRAPPER -------------------
def retry_request(func, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning("Error calling {}: {}. Retry {}/{}".format(func.__name__, e, i+1, retries))
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
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    logger.debug("Requesting historical klines for {} with params: {}", symbol, params)
    response = retry_request(requests.get, url, params=params).json()
    check_response(response)
    data = response["result"]["list"]
    logger.debug("Historical klines raw data for {}: {}", symbol, data)
    if data and isinstance(data[0], list):
        columns = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(data, columns=columns)
    else:
        df = pd.DataFrame(data)
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit='ms')
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.debug("Historical klines dataframe for {} (chronological):\n{}", symbol, df.head())
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
    df['20d_low']  = df['low'].rolling(window=20).min()
    df['10d_high'] = df['high'].rolling(window=10).max()
    df['10d_low']  = df['low'].rolling(window=10).min()
    df['prev_20d_high'] = df['20d_high'].shift(1)
    df['prev_20d_low']  = df['20d_low'].shift(1)
    df['prev_10d_high'] = df['10d_high'].shift(1)
    df['prev_10d_low']  = df['10d_low'].shift(1)
    
    df['long_entry']  = df['close'] > df['prev_20d_high']
    df['long_exit']   = df['close'] < df['prev_10d_low']
    df['short_entry'] = df['close'] < df['prev_20d_low']
    df['short_exit']  = df['close'] > df['prev_10d_high']
    
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

def calculate_position_size(entry_price, stop_loss_price, equity):
    risk_amount = equity * RISK_PERCENT
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0
    pos_size = risk_amount / risk_per_unit
    logger.debug("Calculated position size: {} (Risk: {}, Per Unit: {})", pos_size, risk_amount, risk_per_unit)
    return round(pos_size, 2)

def place_order(symbol, side, order_type, qty, price=None, reduce_only=False):
    try:
        logger.debug("Placing order on {}: side={}, order_type={}, qty={}, price={}, reduce_only={}",
                     symbol, side, order_type, qty, price, reduce_only)
        response = retry_request(trading_session.place_active_order,
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

# Global analytics dictionary to track trade performance
analytics = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_profit": 0.0
}

def record_trade(symbol, direction, entry_price, exit_price, qty):
    """Record a closed trade and update analytics.
       For long trades, profit = (exit - entry)*qty.
       For short trades, profit = (entry - exit)*qty.
    """
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
    logger.info("[Analytics] Total Trades: {}, Wins: {}, Losses: {}, Win Rate: {:.2f}%, Total Profit: {:.2f}, Average Profit: {:.2f}",
                analytics["total_trades"], analytics["winning_trades"], analytics["losing_trades"], win_rate, analytics["total_profit"], avg_profit)

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
    filename = f"plot_signals_{symbol}.png"
    plt.savefig(filename)
    plt.close()
    logger.info("Plot saved to {}", filename)

def trade_symbol(symbol):
    df = get_historical_klines(symbol, INTERVAL, limit=HIST_LIMIT)
    df = calculate_turtle_signals(df)
    
    # Save historical signals to CSV for review
    csv_name = f"signals_{symbol}.csv"
    df.to_csv(csv_name, index=False)
    logger.info("[{}] Saved historical signals to {}", symbol, csv_name)
    
    # Plot signals for visual debugging
    plot_signals(df, symbol)
    
    # Log the last 10 bars with any signal
    signals_df = df[(df["long_entry"]) | (df["long_exit"]) |
                    (df["short_entry"]) | (df["short_exit"])]
    logger.info("[{}] Historical signals (last 10):\n{}", symbol, signals_df.tail(10).to_string(index=False))
    
    current_candle = df.iloc[-1]
    current_price = get_latest_price(symbol)
    equity = get_wallet_equity()
    atr_percent = (current_candle['atr'] / current_candle['close']) if current_candle['close'] != 0 else 0.0
    
    # Adaptive ATR threshold: average ATR% over last 14 bars vs. fixed threshold
    recent_atr_percent = df.iloc[-14:]['atr'].mean() / df.iloc[-14:]['close'].mean()
    adaptive_atr_threshold = max(FIXED_ATR_PERCENT_THRESHOLD, recent_atr_percent)
    
    logger.info("[{}] {} | Price: {} | Equity: {} | ATR%: {:.2%} (Adaptive Threshold: {:.2%}) | RSI: {:.2f}",
                symbol, datetime.datetime.now(datetime.timezone.utc).isoformat(),
                current_price, equity, atr_percent, adaptive_atr_threshold, current_candle['rsi'])
    
    position = get_current_position(symbol)
    
    if position is None:
        # No open position; reset pyramid data if exists
        if symbol in pyramid_data:
            del pyramid_data[symbol]
        if (current_candle['long_entry'] and
            current_candle['rsi'] > RSI_LONG_THRESHOLD and
            atr_percent > adaptive_atr_threshold):
            logger.info("[{}] Confirmed LONG entry signal.", symbol)
            stop_loss_price = current_candle['prev_10d_low']
            entry_price = current_price
            pos_size = calculate_position_size(entry_price, stop_loss_price, equity)
            if pos_size <= 0:
                logger.warning("[{}] Long pos_size <= 0, skipping entry.", symbol)
            else:
                logger.info("[{}] LONG size: {} (Entry: {}, Stop: {})", symbol, pos_size, entry_price, stop_loss_price)
                order_resp = place_order(symbol, side="Buy", order_type="Market", qty=pos_size)
                if order_resp:
                    set_trading_stop(symbol, stop_loss=round(stop_loss_price, 4))
                    pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        elif (current_candle['short_entry'] and
              current_candle['rsi'] < RSI_SHORT_THRESHOLD and
              atr_percent > adaptive_atr_threshold):
            logger.info("[{}] Confirmed SHORT entry signal.", symbol)
            stop_loss_price = current_candle['prev_10d_high']
            entry_price = current_price
            pos_size = calculate_position_size(entry_price, stop_loss_price, equity)
            if pos_size <= 0:
                logger.warning("[{}] Short pos_size <= 0, skipping entry.", symbol)
            else:
                logger.info("[{}] SHORT size: {} (Entry: {}, Stop: {})", symbol, pos_size, entry_price, stop_loss_price)
                order_resp = place_order(symbol, side="Sell", order_type="Market", qty=pos_size)
                if order_resp:
                    set_trading_stop(symbol, stop_loss=round(stop_loss_price, 4))
                    pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        else:
            logger.info("[{}] No confirmed entry signal. Waiting...", symbol)
    else:
        pos_side = position.get("side", "")
        entry_price = float(position["entry_price"])
        pos_qty = float(position["size"])
        # Initialize pyramid data if not present
        if symbol not in pyramid_data:
            pyramid_data[symbol] = {'baseline': entry_price, 'count': 0}
        if pos_side == "Buy":
            # Pyramid logic for long positions
            if current_price > pyramid_data[symbol]['baseline'] * (1 + PYRAMID_THRESHOLD) and pyramid_data[symbol]['count'] < PYRAMID_MAX_COUNT:
                additional_size = calculate_position_size(current_price, current_candle['prev_10d_low'], equity) * PYRAMID_ORDER_FACTOR
                if additional_size > 0:
                    logger.info("[{}] Pyramid condition met for LONG. Adding additional order of size: {}", symbol, additional_size)
                    order_resp = place_order(symbol, side="Buy", order_type="Market", qty=additional_size)
                    if order_resp:
                        pyramid_data[symbol]['baseline'] = current_price
                        pyramid_data[symbol]['count'] += 1
            # Partial exit for long positions
            if current_candle['long_exit']:
                logger.info("[{}] Long exit signal triggered. Executing partial exit.", symbol)
                exit_qty = pos_qty * PARTIAL_EXIT_RATIO
                close_position(symbol, qty=exit_qty)
                record_trade(symbol, "long", entry_price, current_price, exit_qty)
            elif current_price > entry_price * 1.05:
                new_sl = round(current_price * 0.98, 4)
                update_trailing_stop(symbol, new_stop_loss=new_sl)
            else:
                logger.info("[{}] Long position: no exit or trailing update triggered.", symbol)
        elif pos_side == "Sell":
            # Pyramid logic for short positions
            if current_price < pyramid_data[symbol]['baseline'] * (1 - PYRAMID_THRESHOLD) and pyramid_data[symbol]['count'] < PYRAMID_MAX_COUNT:
                additional_size = calculate_position_size(current_price, current_candle['prev_10d_high'], equity) * PYRAMID_ORDER_FACTOR
                if additional_size > 0:
                    logger.info("[{}] Pyramid condition met for SHORT. Adding additional order of size: {}", symbol, additional_size)
                    order_resp = place_order(symbol, side="Sell", order_type="Market", qty=additional_size)
                    if order_resp:
                        pyramid_data[symbol]['baseline'] = current_price
                        pyramid_data[symbol]['count'] += 1
            # Partial exit for short positions
            if current_candle['short_exit']:
                logger.info("[{}] Short exit signal triggered. Executing partial exit.", symbol)
                exit_qty = pos_qty * PARTIAL_EXIT_RATIO
                close_position(symbol, qty=exit_qty)
                record_trade(symbol, "short", entry_price, current_price, exit_qty)
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
        time.sleep(60)

if __name__ == '__main__':
    turtle_trading_bot()
