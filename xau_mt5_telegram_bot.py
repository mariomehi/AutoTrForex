"""
Telegram Bot for automated pattern detection + MT5 (FPMarkets demo) trading
Adapted from original Bybit starter: keeps all pattern detectors and bot commands,
but replaces Bybit API calls with MetaTrader5 (MT5) functions so you can use FPMarkets
demo (MT4/MT5) accounts.

Features:
 - /analizza <SYMBOL> <TIMEFRAME> [autotrade yes/no] -> starts continuous analysis
 - /stop <SYMBOL|ALL> -> stops analysis
 - /list -> lists active analyses
 - ATR-based SL/TP, position sizing by USD risk
 - Pattern detectors: Bullish/Bearish Engulfing, Hammer, Shooting Star, Morning/Evening Star, Pin Bar
 - Generates candle chart on signal (mplfinance) and sends to Telegram
 - Places orders using MetaTrader5 when autotrade=yes (FPMarkets demo supported)

USAGE & DEPLOYMENT (short):
 - Set environment variables: TELEGRAM_TOKEN, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
 - Install requirements: python-telegram-bot, MetaTrader5, pandas, numpy, mplfinance, ta
 - Run the script (preferably on Railway or VPS). The script initializes MT5 connection at start.

IMPORTANT NOTES:
 - This script assumes the broker has XAUUSD symbol (common) and that lots and contract sizes
   are compatible with the simplistic sizing function here. Backtest and paper-trade first.
 - Adjust RISK_USD, ATR multipliers, and ENABLED_TFS to taste.

"""

import os
import time
import math
import logging
import asyncio
from datetime import datetime, timezone
import threading
import tempfile

import MetaTrader5 as mt5
import requests
import pandas as pd
import numpy as np
import mplfinance as mpf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', 'PUT_YOUR_TELEGRAM_TOKEN')
MT5_LOGIN = int(os.environ.get('MT5_LOGIN', '0'))
MT5_PASSWORD = os.environ.get('MT5_PASSWORD', '')
MT5_SERVER = os.environ.get('MT5_SERVER', 'FPMarkets-Demo')

# Strategy parameters (kept as in original)
VOLUME_FILTER = True
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.0
RISK_USD = 10.0
ENABLED_TFS = ['5m','15m','30m','1h','4h']

# Active analyses storage: chat_id -> dict(key->job)
ACTIVE_ANALYSES = {}
ACTIVE_ANALYSES_LOCK = threading.Lock()

# ----------------------------- MT5 HELPERS -----------------------------

TF_MAP = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
}


def mt5_get_klines(symbol: str, timeframe: str, limit: int = 200):
    tf = TF_MAP.get(timeframe)
    if tf is None:
        return pd.DataFrame()

    # Ensure symbol is available
    sym = mt5.symbol_info(symbol)
    if sym is None:
        logging.warning(f"Symbol {symbol} not found in MT5")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    # mt5 returns: open, high, low, close, tick_volume, spread, real_volume
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open','high','low','close','volume']]


# ----------------------------- INDICATORS & PATTERNS -----------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# Pattern detectors (identical to original logic)

def is_bullish_engulfing(prev, curr):
    return (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['open'] <= prev['close'] and curr['close'] >= prev['open'])


def is_bearish_engulfing(prev, curr):
    return (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
            curr['open'] >= prev['close'] and curr['close'] <= prev['open'])


def is_hammer(candle):
    body = abs(candle['close'] - candle['open'])
    lower_wick = candle['open'] - candle['low'] if candle['close'] >= candle['open'] else candle['close'] - candle['low']
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    return lower_wick > 2 * body and upper_wick < body


def is_shooting_star(candle):
    body = abs(candle['close'] - candle['open'])
    upper_wick = candle['high'] - max(candle['open'], candle['close'])
    lower_wick = min(candle['open'], candle['close']) - candle['low']
    return upper_wick > 2 * body and lower_wick < body


def is_morning_star(a, b, c):
    return (a['close'] < a['open'] and
            b['close'] < b['open'] and
            c['close'] > c['open'] and
            c['close'] > (a['open'] + a['close'])/2)


def is_evening_star(a, b, c):
    return (a['close'] > a['open'] and
            b['close'] > b['open'] and
            c['close'] < c['open'] and
            c['close'] < (a['open'] + a['close'])/2)


def is_pin_bar(candle):
    body = abs(candle['close'] - candle['open'])
    upper = candle['high'] - max(candle['close'], candle['open'])
    lower = min(candle['close'], candle['open']) - candle['low']
    return (lower > 2 * body and upper < body) or (upper > 2 * body and lower < body)


def check_patterns(df: pd.DataFrame):
    if len(df) < 4:
        return (False, None, None)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    if is_bullish_engulfing(prev, last):
        return (True, 'Buy', 'Bullish Engulfing')
    if is_bearish_engulfing(prev, last):
        return (True, 'Sell', 'Bearish Engulfing')
    if is_hammer(last):
        return (True, 'Buy', 'Hammer')
    if is_shooting_star(last):
        return (True, 'Sell', 'Shooting Star')
    if is_morning_star(prev2, prev, last):
        return (True, 'Buy', 'Morning Star')
    if is_evening_star(prev2, prev, last):
        return (True, 'Sell', 'Evening Star')
    if is_pin_bar(last):
        side = 'Buy' if (min(last['open'], last['close']) - last['low']) > (last['high'] - max(last['open'], last['close'])) else 'Sell'
        return (True, side, 'Pin Bar')
    return (False, None, None)


# ----------------------------- POSITION SIZING -----------------------------

def calculate_position_size(symbol: str, entry_price: float, sl_price: float, risk_usd: float):
    """
    Approximate sizing for MT5: convert USD risk to lots.
    This simple function assumes: 1 lot = 100 oz for XAUUSD (varies by broker). Adjust if needed.
    """
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0.0
    # assume 1 lot = 100 units (typical for many brokers for XAUUSD), and contract currency is USD
    units_per_lot = 100
    risk_per_lot = risk_per_unit * units_per_lot
    lots = risk_usd / risk_per_lot
    # round to 2 decimals (typical lot step 0.01)
    lots = max(0.0, round(lots, 2))
    return lots


# ----------------------------- MT5 ORDER (synchronous) -----------------------------

def place_mt5_order(symbol, side, lots, sl, tp):
    info = mt5.symbol_info(symbol)
    if info is None:
        return {'ret': 'error', 'error': 'symbol_not_found'}

    # ensure symbol is visible
    if not info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {'ret': 'error', 'error': 'no_tick'}

    if side == 'Buy':
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': float(lots),
        'type': order_type,
        'price': price,
        'sl': float(sl) if sl is not None else 0.0,
        'tp': float(tp) if tp is not None else 0.0,
        'deviation': 20,
        'magic': 123456,
        'comment': 'XAU Bot MT5',
        'type_filling': mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    return result._asdict() if hasattr(result, '_asdict') else result


# ----------------------------- JOB CALLBACK -----------------------------

async def analyze_job(context: ContextTypes.DEFAULT_TYPE):
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']

    try:
        df = mt5_get_klines(symbol, timeframe, limit=200)
        if df.empty:
            await context.bot.send_message(chat_id=chat_id, text=f'No data for {symbol} {timeframe}')
            return

        # Apply volume filter
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                if vol.iloc[-1] <= vol.iloc[-21:-1].mean():
                    return

        found, side, pattern = check_patterns(df)
        if not found:
            return

        # Calculate ATR-based SL/TP
        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else np.nan
        last_close = df['close'].iloc[-1]

        sl_price = None
        tp_price = None
        if not math.isnan(last_atr):
            if side == 'Buy':
                sl_price = last_close - last_atr * ATR_MULT_SL
                tp_price = last_close + last_atr * ATR_MULT_TP
            else:
                sl_price = last_close + last_atr * ATR_MULT_SL
                tp_price = last_close - last_atr * ATR_MULT_TP
        else:
            if side == 'Buy':
                sl_price = df['low'].iloc[-1]
                tp_price = last_close * (1 + 0.02)
            else:
                sl_price = df['high'].iloc[-1]
                tp_price = last_close * (1 - 0.02)

        # position sizing based on USD risk (converted to lots)
        qty_lots = calculate_position_size(symbol, last_close, sl_price, RISK_USD)
        if qty_lots <= 0:
            await context.bot.send_message(chat_id=chat_id, text=f'Calculated lots is 0 for {symbol}. Check parameters.')
            return

        # Generate chart and send
        tmpf = tempfile.gettempdir()
        chart_path = os.path.join(tmpf, f'{symbol}_{timeframe}_{int(time.time())}.png')
        mpf.plot(df.tail(100), type='candle', style='charles', savefig=chart_path)

        caption = f"Signal: {pattern} \nSide: {side}\nSymbol: {symbol} {timeframe}\nPrice: {last_close:.4f}\nSL: {sl_price:.4f}\nTP: {tp_price:.4f}\nLots(approx): {qty_lots:.2f}"
        await context.bot.send_photo(chat_id=chat_id, photo=open(chart_path, 'rb'), caption=caption)

        # Place order on MT5 (only if enabled by job context 'autotrade')
        if job_ctx.get('autotrade'):
            # MT5 order_send is synchronous -> run in executor
            loop = asyncio.get_running_loop()
            order_res = await loop.run_in_executor(None, place_mt5_order, symbol, side, qty_lots, sl_price, tp_price)
            await context.bot.send_message(chat_id=chat_id, text=f'Order result: {order_res}')

    except Exception as e:
        logging.exception('analyze_job error')
        await context.bot.send_message(chat_id=chat_id, text=f"Errore nell'analisi di {symbol} {timeframe}: {e}")


# ----------------------------- TELEGRAM COMMANDS -----------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Bot attivo. Usa /analizza <SYMBOL> <TIMEFRAME> per iniziare. Esempio: /analizza XAUUSD 15m')


async def cmd_analizza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text('Usage: /analizza SYMBOL TIMEFRAME [autotrade yes/no]')
        return
    symbol = args[0].upper()
    timeframe = args[1]
    autotrade = (len(args) > 2 and args[2].lower() in ['yes', 'true', '1'])

    if timeframe not in ENABLED_TFS:
        await update.message.reply_text(f'Timeframe non supportato. Abilitati: {ENABLED_TFS}')
        return

    key = f'{symbol}-{timeframe}'
    with ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.setdefault(chat_id, {})
        if key in chat_map:
            await update.message.reply_text(f'Già analizzando {symbol} {timeframe} in questa chat.')
            return
        interval_seconds = int({'5m':300,'15m':900,'30m':1800,'1h':3600,'4h':14400}[timeframe])
        now = datetime.now(timezone.utc)
        epoch = int(now.timestamp())
        to_next = interval_seconds - (epoch % interval_seconds)

        job_data = {'chat_id': chat_id, 'symbol': symbol, 'timeframe': timeframe, 'autotrade': autotrade}
        job = context.job_queue.run_repeating(analyze_job, interval=interval_seconds, first=to_next, data=job_data)
        chat_map[key] = job

    await update.message.reply_text(f'Iniziata analisi {symbol} {timeframe}. Autotrade: {autotrade}')


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 1:
        await update.message.reply_text('Usage: /stop SYMBOL or /stop all')
        return
    target = args[0].upper()
    with ACTIVE_ANALYSES_LOCK:
        chat_map = ACTIVE_ANALYSES.get(chat_id, {})
        if target == 'ALL':
            for k, job in list(chat_map.items()):
                job.schedule_removal()
                del chat_map[k]
            await update.message.reply_text('Tutte le analisi fermate.')
            return
        removed = False
        for k in list(chat_map.keys()):
            if k.startswith(target + '-'):
                job = chat_map[k]
                job.schedule_removal()
                del chat_map[k]
                removed = True
        if removed:
            await update.message.reply_text(f'Analisi per {target} fermata.')
        else:
            await update.message.reply_text(f'Non trovata analisi attiva per {target} in questa chat.')


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_map = ACTIVE_ANALYSES.get(chat_id, {})
    if not chat_map:
        await update.message.reply_text('Nessuna analisi attiva in questa chat.')
        return
    text = 'Analisi attive:\n' + '\n'.join(chat_map.keys())
    await update.message.reply_text(text)


# ----------------------------- MAIN -----------------------------

def main():
    logging.basicConfig(level=logging.INFO)

    if TELEGRAM_TOKEN.startswith('PUT_YOUR'):
        logging.error('Set TELEGRAM_TOKEN environment variable before running')
        return

    # Initialize MT5
    if not mt5.initialize():
        logging.error('Errore inizializzazione MT5')
        return

    if MT5_LOGIN and MT5_PASSWORD:
        logged = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not logged:
            logging.error('Login MT5 fallito. Controlla credenziali e server.')
            return
        else:
            logging.info('Login MT5 riuscito')

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CommandHandler('analizza', cmd_analizza))
    application.add_handler(CommandHandler('stop', cmd_stop))
    application.add_handler(CommandHandler('list', cmd_list))

    logging.info('Starting bot...')
    application.run_polling()


if __name__ == '__main__':
    main()
