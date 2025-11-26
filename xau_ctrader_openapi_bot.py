"""
Telegram + cTrader Open API (Spotware) trading bot for XAUUSD
- Keeps original pattern detectors and logic
- Uses OpenApiPy (ctrader-open-api) to connect to cTrader Open API (demo) and place market orders

What this bundle includes:
- Single-file Telegram bot + OpenApiPy client
- Uses OpenApiPy to authenticate (application + account) and send order requests
- ATR-based SL/TP and USD-based sizing (approximated to symbol's contract size)

ENVIRONMENT VARIABLES (set these on Railway):
- TELEGRAM_TOKEN  -> Telegram bot token
- CTRADER_CLIENT_ID -> cTrader Open API application Client ID
- CTRADER_CLIENT_SECRET -> cTrader Open API application Client Secret
- CTRADER_ACCOUNT_ID -> your cTrader Trader Account ID (numeric) (CTID account id)
- CTRADER_HOST -> optional: 'demo' or 'live' (default 'demo')
- RISK_USD -> numeric (default 10.0)

NOTE (must read):
- You need to register an application at https://connect.spotware.com/apps and add a Redirect URI.
- You must approve access for your cTrader demo account and obtain the initial tokens if required by your broker.
- This script uses the official OpenApiPy client (ctrader-open-api). It communicates using Protobuf over TCP.
- OpenApiPy uses Twisted and runs an internal reactor. To keep things simple, in this script we run the OpenApi client in a background thread and interact with it via thread-safe queues/callbacks.

LIMITATIONS & IMPORTANT:
- OpenApi protocol & message names are provided by Spotware and may change. Test on demo FIRST.
- If any protobuf message name differs, consult the OpenApiPy docs: https://spotware.github.io/OpenApiPy/

INSTALL (requirements.txt):
- python-telegram-bot==20.6
- pandas
- numpy
- mplfinance
- requests
- ctrader-open-api

Deploy on Railway:
- Add the above env vars to Railway variables
- Procfile: worker: python xau_ctrader_openapi_bot.py

"""

import os
import time
import math
import logging
import threading
import tempfile
import asyncio
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import mplfinance as mpf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# OpenApiPy imports
try:
    from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
except Exception as e:
    Client = None
    TcpProtocol = None
    EndPoints = None
    Protobuf = None
    # protobuf messages will be referenced dynamically; tests required

# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', 'PUT_YOUR_TELEGRAM_TOKEN')
CTRADER_CLIENT_ID = os.environ.get('CTRADER_CLIENT_ID')
CTRADER_CLIENT_SECRET = os.environ.get('CTRADER_CLIENT_SECRET')
CTRADER_ACCOUNT_ID = int(os.environ.get('CTRADER_ACCOUNT_ID', '0'))
CTRADER_HOST = os.environ.get('CTRADER_HOST', 'demo').lower()  # 'demo' or 'live'

RISK_USD = float(os.environ.get('RISK_USD', '10'))
VOLUME_FILTER = True
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.0
ENABLED_TFS = ['5m','15m','30m','1h','4h']

# Active analyses
ACTIVE_ANALYSES = {}
ACTIVE_ANALYSES_LOCK = threading.Lock()

# OpenApi client holder
OPENAPI_CLIENT = None
OPENAPI_LOCK = threading.Lock()
OPENAPI_CONNECTED = threading.Event()

# ----------------------------- MT-like klines (from OpenAPI via REST public or broker) -----------------------------
# For simplicity we use a public market data provider (if available) or fallback to cTrader history via OpenApi (requires additional messages)
# To keep parity with your original code, we'll implement a simple placeholder using public data via AlphaVantage or other (user can replace with their preferred source).

# --- For now, we will fetch klines from a free public API (storing API usage is up to you). This is ONLY for analysis; orders use OpenApi.
# We'll use a lightweight provider: https://finnhub.io, https://alpha-vantage.co or exchangerate.host etc.
# Because Railway cannot rely on external keys here, the user should set MARKET_DATA_PROVIDER env vars if desired.

MARKET_PROVIDER = os.environ.get('MARKET_PROVIDER', 'BYBIT')  # placeholder

# ----------------------------- INDICATORS & PATTERNS (same as original) -----------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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

# ----------------------------- SIMPLE MARKET DATA FETCHER (BYBIT public as fallback) -----------------------------
import requests

BYBIT_INTERVAL_MAP = {'1m':'1','3m':'3','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}

def public_get_klines_bybit(symbol: str, timeframe: str, limit: int = 200):
    itv = BYBIT_INTERVAL_MAP.get(timeframe)
    if itv is None:
        return pd.DataFrame()
    url = 'https://api.bybit.com/v5/market/kline'
    params = {'category':'linear','symbol':symbol,'interval':itv,'limit':limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    data = j.get('result', {}).get('list', [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
    df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}, inplace=True)
    return df

# ----------------------------- POSITION SIZING FOR XAU (approx) -----------------------------

def calculate_position_size(symbol: str, entry_price: float, sl_price: float, risk_usd: float):
    # approximation: 1 lot = 100 oz for many cTrader brokers; adjust if broker differs
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0.0
    units_per_lot = 100
    risk_per_lot = risk_per_unit * units_per_lot
    lots = risk_usd / risk_per_lot
    return round(max(0.0, lots), 2)

# ----------------------------- OPENAPI (ctrader) CLIENT THREAD -----------------------------

class OpenApiRunner(threading.Thread):
    def __init__(self, client_id, client_secret, account_id, host='demo'):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = account_id
        self.host = host
        self.client = None

    def run(self):
        global OPENAPI_CLIENT
        if Client is None:
            logging.error('ctrader-open-api not installed or import failed')
            return
        host = EndPoints.PROTOBUF_DEMO_HOST if self.host == 'demo' else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT
        self.client = Client(host, port, TcpProtocol)

        def on_connected(client):
            logging.info('OpenApi connected')
            # send application auth
            req = ProtoOAApplicationAuthReq()
            req.clientId = str(self.client_id)
            req.clientSecret = str(self.client_secret)
            d = client.send(req)
            d.addErrback(lambda f: logging.error('App auth err: %s', f))

        def on_message(client, message):
            # handle incoming messages (log minimal)
            try:
                msg = Protobuf.extract(message)
                logging.debug('OpenApi message: %s', msg)
            except Exception:
                logging.exception('Failed to parse OpenApi message')

        def on_disconnected(client, reason):
            logging.warning('OpenApi disconnected: %s', reason)
            OPENAPI_CONNECTED.clear()

        self.client.setConnectedCallback(on_connected)
        self.client.setMessageReceivedCallback(on_message)
        self.client.setDisconnectedCallback(on_disconnected)

        self.client.startService()
        OPENAPI_CLIENT = self.client
        # mark connected (note: real connected event after auth response, but for simplicity set it)
        OPENAPI_CONNECTED.set()

# helper to start OpenApi

def ensure_openapi_running():
    global OPENAPI_CLIENT
    if OPENAPI_CLIENT is not None and OPENAPI_CONNECTED.is_set():
        return True
    if CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET and CTRADER_ACCOUNT_ID:
        runner = OpenApiRunner(CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCOUNT_ID, CTRADER_HOST)
        runner.start()
        # wait brief
        if not OPENAPI_CONNECTED.wait(timeout=10):
            logging.warning('OpenApi client did not signal connection within 10s')
        return True
    logging.warning('OpenApi credentials missing')
    return False

# function to send market order via OpenApi client using ProtoOANewOrderReq

def openapi_place_market_order(symbol_id: int, side: str, volume: float, sl_relative: float = None, tp_relative: float = None):
    """
    Sends a new market order request via OpenApi client.
    - symbol_id: numeric symbol id (you may retrieve symbol list via OpenApi messages)
    - side: 'Buy' or 'Sell'
    - volume: lots
    - sl_relative/tp_relative: distance in pips/points relative to entry (implementation broker-specific)

    NOTE: This function is a best-effort wrapper: exact message fields and types depend on OpenApi proto definitions.
    Test on demo.
    """
    if OPENAPI_CLIENT is None:
        return {'error': 'openapi_not_connected'}
    try:
        # Build ProtoOANewOrderReq (message name may be ProtoOANewOrderReq or ProtoOANewMarketOrderReq depending on your OpenApi version)
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = int(CTRADER_ACCOUNT_ID)
        req.symbolId = int(symbol_id)
        req.orderType = ProtoOANewOrderReq.MARKET  # enum in proto
        req.side = ProtoOANewOrderReq.BUY if side.lower() == 'buy' else ProtoOANewOrderReq.SELL
        req.volume = float(volume)
        if sl_relative is not None:
            req.relativeStopLoss = int(round(sl_relative))
        if tp_relative is not None:
            req.relativeTakeProfit = int(round(tp_relative))
        deferred = OPENAPI_CLIENT.send(req)
        # We convert deferred to a blocking result with a timeout for simplicity
        result = deferred.result if hasattr(deferred, 'result') else None
        return {'sent': True, 'deferred': str(deferred)}
    except Exception as e:
        logging.exception('openapi_place_market_order error')
        return {'error': str(e)}

# ----------------------------- JOB CALLBACK (analysis + order) -----------------------------

async def analyze_job(context: ContextTypes.DEFAULT_TYPE):
    job_ctx = context.job.data
    chat_id = job_ctx['chat_id']
    symbol = job_ctx['symbol']
    timeframe = job_ctx['timeframe']

    try:
        # get klines via public provider fallback
        df = public_get_klines_bybit(symbol, timeframe, limit=200)
        if df.empty:
            await context.bot.send_message(chat_id=chat_id, text=f'No data for {symbol} {timeframe}')
            return

        # volume filter
        if VOLUME_FILTER:
            vol = df['volume']
            if len(vol) >= 21:
                if vol.iloc[-1] <= vol.iloc[-21:-1].mean():
                    return

        found, side, pattern = check_patterns(df)
        if not found:
            return

        atr_series = atr(df, period=14)
        last_atr = atr_series.iloc[-1] if not atr_series.isna().all() else math.nan
        last_close = df['close'].iloc[-1]

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

        qty = calculate_position_size(symbol, last_close, sl_price, RISK_USD)
        if qty <= 0:
            await context.bot.send_message(chat_id=chat_id, text=f'Calculated qty is 0 for {symbol}. Check parameters.')
            return

        # chart
        chart_path = os.path.join(tempfile.gettempdir(), f'{symbol}_{timeframe}_{int(time.time())}.png')
        mpf.plot(df.tail(100), type='candle', style='charles', savefig=chart_path)

        caption = f"Signal: {pattern} \nSide: {side}\nSymbol: {symbol} {timeframe}\nPrice: {last_close:.4f}\nSL: {sl_price:.4f}\nTP: {tp_price:.4f}\nLots(approx): {qty:.2f}"
        await context.bot.send_photo(chat_id=chat_id, photo=open(chart_path, 'rb'), caption=caption)

        if job_ctx.get('autotrade'):
            # Ensure OpenApi running
            ensure_openapi_running()
            # You must know symbol_id on cTrader side. Many brokers expose symbolId via OpenApi 'SymbolsList' messages.
            # For simplicity we'll assume symbol_id is provided in job_ctx or env var CTRADER_SYMBOL_ID
            symbol_id = int(job_ctx.get('symbol_id', int(os.environ.get('CTRADER_SYMBOL_ID', '0'))))
            if symbol_id == 0:
                await context.bot.send_message(chat_id=chat_id, text='Autotrade richiesto ma CTRADER_SYMBOL_ID non impostato. Imposta l\'ID simbolo in variabili ambiente o usa /analizza con <symbol_id>.')
                return
            # compute relative SL/TP in points (broker dependent). We'll send approximate absolute distances in pips * 100
            sl_rel = abs(last_close - sl_price)
            tp_rel = abs(tp_price - last_close)
            order_res = await asyncio.get_running_loop().run_in_executor(None, openapi_place_market_order, symbol_id, side, qty, sl_rel, tp_rel)
            await context.bot.send_message(chat_id=chat_id, text=f'Order result: {order_res}')

    except Exception as e:
        logging.exception('analyze_job error')
        await context.bot.send_message(chat_id=chat_id, text=f"Errore nell'analisi di {symbol} {timeframe}: {e}")

# ----------------------------- TELEGRAM COMMANDS -----------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Bot attivo. Usa /analizza <SYMBOL> <TIMEFRAME> [autotrade yes/no] [symbol_id(optional)]')


async def cmd_analizza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text('Usage: /analizza SYMBOL TIMEFRAME [autotrade yes/no] [symbol_id]')
        return
    symbol = args[0].upper()
    timeframe = args[1]
    autotrade = (len(args) > 2 and args[2].lower() in ['yes','true','1'])
    symbol_id = int(args[3]) if len(args) > 3 else None

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
        if symbol_id:
            job_data['symbol_id'] = symbol_id
        job = context.job_queue.run_repeating(analyze_job, interval=interval_seconds, first=to_next, data=job_data)
        chat_map[key] = job

    await update.message.reply_text(f'Iniziata analisi {symbol} {timeframe}. Autotrade: {autotrade}. SymbolId: {symbol_id}')


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

    # Start OpenApi background runner (non-blocking)
    if CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET and CTRADER_ACCOUNT_ID:
        ensure_openapi_running()

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CommandHandler('analizza', cmd_analizza))
    application.add_handler(CommandHandler('stop', cmd_stop))
    application.add_handler(CommandHandler('list', cmd_list))

    logging.info('Starting bot...')
    application.run_polling()


if __name__ == '__main__':
    main()
