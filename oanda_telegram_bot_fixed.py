"""
Telegram Bot for automated pattern detection + OANDA Practice trading (Forex)
- Features:
  * /analizza <SYMBOL> <TIMEFRAME> -> starts continuous analysis for that symbol+tf
  * /stop <SYMBOL> -> stops analysis for that symbol (in that chat)
  * multi-symbol, multi-timeframe per chat
  * volume filter
  * SL = ATR * X (user config)
  * TP = ATR * X (user config)
  * position sizing by risk per trade (USD risk)
  * uses OANDA Practice for orders and klines (oandapyV20)
  * generates candle chart when signal is found (mplfinance)

Notes:
- Questa è la versione modificata. Devi impostare il tuo TELEGRAM_BOT_TOKEN e le chiavi API OANDA (ACCESS_TOKEN e ACCOUNT_ID).
- Testare a fondo sull'ambiente PRACTICE di OANDA prima di considerare un account LIVE.
- Progettato per essere eseguito su Railway / VPS.

Telegram Bot per pattern detection + OANDA Trading - VERSIONE MODIFICATA
Adattato da Bybit a OANDA (Forex/CFD)
Correzioni principali:
- Rimpiazzato 'pybit' con 'oandapyV20'.
- Nuove funzioni per la connessione, dati e ordini OANDA.
- Gestione corretta del filesystem su Railway
- Matplotlib backend non-GUI
- Migliore gestione errori
- Logging migliorato
"""

import os
import time
import math
import logging
from datetime import datetime, timezone
import threading
import io
import tempfile

# IMPORTANTE: Configura matplotlib prima di altri import
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI per server

import requests
import pandas as pd
import numpy as np
import mplfinance as mpf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import telegram.error

# Import OANDA V20
try:
    import oandapyV20
    import oandapyV20.endpoints.accounts as accounts
    import oandapyV20.endpoints.instruments as instruments
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.positions as positions
    # Import componenti per richieste ordine con SL/TP
    from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
    OANDA_API = oandapyV20.API
    OANDA_V20_AVAILABLE = True
except Exception as e:
    logging.warning(f'oandapyV20 import failed: {e}. Installa: pip install oandapyV20')
    OANDA_V20_AVAILABLE = False
    OANDA_API = None


# ----------------------------- CONFIG -----------------------------
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
# --- NUOVA CONFIGURAZIONE OANDA ---
OANDA_ACCESS_TOKEN = os.environ.get('OANDA_ACCESS_TOKEN', '')
OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID', '')
# ----------------------------------

# Scegli l'ambiente di trading
# 'practice' = Demo Trading (fondi virtuali su OANDA)
# 'live' = Trading Reale (ATTENZIONE: soldi veri!)
TRADING_MODE = os.environ.get('TRADING_MODE', 'practice')

# Strategy parameters
VOLUME_FILTER = True
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.0
RISK_USD = 10.0
ENABLED_TFS = ['1m','3m','5m','15m','30m','1h','4h']

# Symbol-specific risk overrides (opzionale)
SYMBOL_RISK_OVERRIDE = {
    # Esempio: per EUR_USD usa solo $5 invece di $10
    # 'EUR_USD': 5.0,
}

# Klines map (Bybit v5 -> OANDA Granularity)
OANDA_GRANULARITY_MAP = {
    '1m': 'M1', '3m': 'M3', '5m': 'M5', '15m': 'M15', '30m': 'M30', 
    '1h': 'H1', '4h': 'H4'
}
# OANDA supporta anche H8, D, W, ecc.
# Nota: La OANDA API lavora con M1, M5, H1, ecc. per le granularità.

# Interval to seconds mapping (rimane invariato)
INTERVAL_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900, 
    '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400
}

# Active analyses storage
ACTIVE_ANALYSES = {}
ACTIVE_ANALYSES_LOCK = threading.Lock()

# Paused notifications: chat_id -> set of "SYMBOL-TIMEFRAME" keys
PAUSED_NOTIFICATIONS = {}
PAUSED_LOCK = threading.Lock()

# Active positions tracking: symbol -> order_info
ACTIVE_POSITIONS = {}
POSITIONS_LOCK = threading.Lock()


def create_oanda_api():
    """Crea sessione OANDA per trading (Practice o Live)"""
    if not OANDA_V20_AVAILABLE:
        raise RuntimeError('oandapyV20 non disponibile. Installa: pip install oandapyV20')
    if not OANDA_ACCESS_TOKEN or not OANDA_ACCOUNT_ID:
        raise RuntimeError('OANDA_ACCESS_TOKEN e OANDA_ACCOUNT_ID devono essere configurate')
    
    # Determina l'ambiente in base alla modalità
    environment = 'practice' if TRADING_MODE == 'practice' else 'live'
    
    logging.info(f'🔌 Connessione OANDA - Modalità: {TRADING_MODE.upper()}')
    logging.info(f'📡 Ambiente: {environment}')
    
    # Inizializza il client API di OANDA
    api = OANDA_API(
        access_token=OANDA_ACCESS_TOKEN,
        environment=environment
    )
    
    return api

# ----------------------------- UTILITIES -----------------------------

def oanda_get_klines(symbol: str, interval: str, count: int = 200):
    """
    Ottiene klines (candele) da OANDA V20 API
    Returns: DataFrame con OHLCV
    """
    granularity = OANDA_GRANULARITY_MAP.get(interval)
    if granularity is None:
        raise ValueError(f'Timeframe OANDA non supportato: {interval}')

    instrument = symbol # Assumiamo che il SYMBOL sia nel formato OANDA (es. EUR_USD)
    
    try:
        # Usa l'endpoint InstrumentsCandles
        params = {
            'count': count,
            'granularity': granularity,
            'price': 'M' # Prezzi Midpoint
        }
        
        # Crea e esegui la richiesta
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        api = create_oanda_api()
        response = api.request(r)
        
        data = response.get('candles', [])
        if not data:
            logging.warning(f'Nessun dato per {instrument} {interval}')
            return pd.DataFrame()

        # OANDA restituisce i dati dal più vecchio al più recente (perfetto)
        df = pd.DataFrame([
            {
                'timestamp': item['time'],
                'open': float(item['mid']['o']),
                'high': float(item['mid']['h']),
                'low': float(item['mid']['l']),
                'close': float(item['mid']['c']),
                'volume': float(item.get('volume', 0)),
                'complete': item.get('complete', False)
            } for item in data if item.get('complete') # Importante: prendi solo le candele complete
        ])
        
        if df.empty:
             logging.warning(f'Nessuna candela completa per {instrument} {interval}')
             return pd.DataFrame()
             
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Le colonne 'volume' in OANDA sono i tick count, non volumi reali, ma le manteniamo.
        df = df[['open','high','low','close','volume']].astype(float)
        
        return df
        
    except oandapyV20.exceptions.V20Error as e:
        logging.error(f'Errore OANDA V20 API (klines): {e}. Status: {e.response.status_code}')
        return pd.DataFrame()
    except Exception as e:
        logging.error(f'Errore nel parsing klines OANDA: {e}')
        return pd.DataFrame()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcola Average True Range (Logica invariata)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.ewm(span=period, adjust=False).mean() # Utilizzo EMA per ATR come comune


# ----------------------------- PATTERN DETECTION -----------------------------

# ... (Le funzioni di rilevamento pattern 'is_bullish_engulfing', ecc., rimangono invariate) ...

# ----------------------------- TRADING LOGIC -----------------------------

# --- Riscritto per OANDA ---
def oanda_format_api_error(error_code: int, error_msg: str) -> str:
    """Formatta un messaggio di errore API OANDA per Telegram."""
    msg = f"❌ <b>Errore API OANDA</b>\n\n"
    msg += f"Codice HTTP: {error_code}\n"
    msg += f"Messaggio: {error_msg}\n\n"
    
    if error_code in [401, 403]: 
        msg += "💡 API Key/Token non valido o scaduto, o AccountID errato.\n"
        msg += "Soluzione: Verifica OANDA_ACCESS_TOKEN e OANDA_ACCOUNT_ID."
    elif error_code == 400:
        msg += "💡 Errore di richiesta (es. strumento non valido, quantità non consentita).\n"
        msg += "Soluzione: Controlla il SYMBOL e i parametri dell'ordine."
    else:
        msg += "Controlla il log per i dettagli."
        
    return msg

# --- Funzione calculate_position_size invariata nel principio ---
def calculate_position_size(entry_price: float, sl_price: float, risk_usd: float):
    """
    Calcola la quantità basata sul rischio in USD
    Formula: Qty (in unità di base) = Risk USD / |Entry - SL|
    """
    risk_per_unit = abs(entry_price - sl_price)
    
    if risk_per_unit > 0:
        # La quantità qui è in unità di base (es. EUR per EUR_USD)
        qty = risk_usd / risk_per_unit
        
        # Per OANDA (Forex), l'unità minima è tipicamente 1 o 1000.
        # Riduciamo la precisione a 0 decimali per operare con unità intere, 
        # a meno che l'utente non operi con conti mini/micro che supportano decimali.
        # Assumiamo unità intere per semplicità (es. 10000)
        qty = math.floor(qty) 
        
        # Limite minimo/massimo sensato per l'esempio (regolare in base al conto OANDA)
        if qty < 1:
            logging.warning(f'Quantità calcolata {qty} troppo piccola. Imposta a 0.')
            return 0
        
        return qty

    else:
        logging.warning(f'Rischio per unità zero. Impossibile calcolare la quantità.')
        return 0

# --- Riscritto per OANDA ---
def place_oanda_order(symbol: str, side: str, qty: float, entry_price: float, sl_price: float, tp_price: float) -> dict:
    """
    Piazza un ordine Market su OANDA con SL/TP.
    Restituisce un dizionario con i dettagli dell'ordine.
    """
    if not OANDA_V20_AVAILABLE:
        logging.error('oandapyV20 non disponibile per piazzare ordini.')
        return {'success': False, 'msg': 'API non disponibile'}
        
    # Le unità sono positive per Long (Buy) e negative per Short (Sell).
    units = int(qty) if side.lower() == 'buy' else int(-qty)
    
    if units == 0:
        logging.warning(f"Quantità calcolata è zero ({qty}) per {symbol} ({side}). Ordine non piazzato.")
        return {'success': False, 'msg': 'Quantità calcolata zero'}
    
    # Dettagli Stop Loss e Take Profit (usiamo 5 decimali per FX)
    sl_details = StopLossDetails(price=f"{sl_price:.5f}").data
    tp_details = TakeProfitDetails(price=f"{tp_price:.5f}").data
    
    # Crea la richiesta di ordine Market
    try:
        # NOTA: OANDA V20 API non permette l'inserimento dell'Entry Price
        # per un ordine Market, usa il prezzo di mercato corrente.
        mkt_order = MarketOrderRequest(
            instrument=symbol,
            units=units,
            timeInForce='FOK',  # Fill or Kill (esegui immediatamente o annulla)
            stopLossOnFill=sl_details,
            takeProfitOnFill=tp_details
        ).data
        
        # Crea la richiesta API
        r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=mkt_order)
        
        api = create_oanda_api()
        response = api.request(r)
        
        # Verifica la risposta e estrai i dettagli del trade
        if 'orderFillTransaction' in response and 'tradeOpened' in response['orderFillTransaction']:
            fill_trans = response['orderFillTransaction']
            trade_opened = fill_trans['tradeOpened']
            
            order_info = {
                'symbol': symbol,
                'side': side.upper(),
                'qty': abs(float(fill_trans['units'])),
                'entry_price': float(fill_trans.get('price', entry_price)),
                'order_id': trade_opened['tradeID'], # L'ID del trade
                'sl': sl_price,
                'tp': tp_price,
                'timestamp': fill_trans['time']
            }
            logging.info(f"✅ Ordine Market OANDA piazzato con successo. Trade ID: {order_info['order_id']}")
            return {'success': True, 'order_info': order_info, 'raw_response': response}
        else:
            # Se l'ordine è stato respinto o non eseguito immediatamente
            msg = response.get('orderRejectTransaction', {}).get('reason', 'Ordine respinto/non eseguito')
            logging.error(f"❌ Errore OANDA nell'esecuzione dell'ordine: {msg}")
            return {'success': False, 'msg': f"Ordine OANDA non eseguito: {msg}", 'raw_response': response}


    except oandapyV20.exceptions.V20Error as e:
        error_msg = str(e)
        error_code = e.response.status_code if hasattr(e.response, 'status_code') else 0
        formatted_error = oanda_format_api_error(error_code, error_msg)
        logging.error(f"❌ Errore OANDA V20 API (order): {formatted_error}")
        return {'success': False, 'msg': formatted_error}
    except Exception as e:
        logging.error(f"❌ Errore generico nel piazzare l'ordine OANDA: {e}")
        return {'success': False, 'msg': f"Errore generico: {e}"}

# --- Riscritto per OANDA ---
async def close_oanda_trade(symbol: str, trade_id: str):
    """Chiude un trade specifico su OANDA."""
    if not OANDA_V20_AVAILABLE:
        return {'success': False, 'msg': 'API non disponibile'}
    
    try:
        # Usa l'endpoint per chiudere una posizione
        # NOTA: Per chiudere un trade specifico si usa l'endpoint trades.TradeClose
        # Ma per ora, per semplicità e in mancanza del trade_id nella nostra struttura,
        # replichiamo la logica di chiusura parziale/totale sulla posizione, 
        # in base al lato (buy/sell).

        # Usiamo TradeClose per chiudere TUTTE le unità del trade
        r = orders.OrderCreate(
            accountID=OANDA_ACCOUNT_ID, 
            data={
                "order": {
                    "type": "MARKET_IF_TOUCHED", # Placeholder, non usiamo qui
                }
            }
        )
        
        # Per chiudere una posizione si usa l'endpoint PositionClose
        # Ma la logica di /chiudi nel codice originale non è completa.
        # Qui implementiamo la chiusura della POSIZIONE.
        
        # Per chiudere tutte le posizioni di un dato strumento
        r = positions.PositionClose(
            accountID=OANDA_ACCOUNT_ID, 
            instrument=symbol, 
            data={
                "longUnits": "ALL",
                "shortUnits": "ALL"
            }
        )
        
        api = create_oanda_api()
        response = api.request(r)
        
        if 'longOrderFillTransaction' in response or 'shortOrderFillTransaction' in response:
            return {'success': True, 'msg': f'Posizione {symbol} chiusa con successo.', 'raw_response': response}
        
        return {'success': False, 'msg': 'Nessuna transazione di chiusura trovata.', 'raw_response': response}

    except oandapyV20.exceptions.V20Error as e:
        error_msg = str(e)
        error_code = e.response.status_code if hasattr(e.response, 'status_code') else 0
        formatted_error = oanda_format_api_error(error_code, error_msg)
        return {'success': False, 'msg': formatted_error}
    except Exception as e:
        logging.error(f'Errore generico in close_oanda_trade: {e}')
        return {'success': False, 'msg': f'Errore generico: {e}'}


# --- Riscritto per OANDA ---
async def get_open_positions_from_oanda():
    """Ottiene le posizioni aperte da OANDA (per la sincronizzazione)."""
    if not OANDA_V20_AVAILABLE:
        return []
        
    try:
        api = create_oanda_api()
        r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
        response = api.request(r)
        
        oanda_positions = response.get('positions', [])
        
        formatted_positions = []
        for p in oanda_positions:
            instrument = p['instrument']
            
            # OANDA V20 ha una posizione "long" e una "short"
            if float(p.get('long', {}).get('units', 0)) > 0:
                side = 'BUY'
                units = float(p['long']['units'])
                price = float(p['long']['averagePrice'])
                # OANDA non espone TradeID qui, usiamo l'instrument
                trade_id = f"POS-{instrument}-LONG" 
                
                formatted_positions.append({
                    'symbol': instrument,
                    'side': side,
                    'qty': units,
                    'entry_price': price,
                    'order_id': trade_id, 
                    'pnl': float(p.get('unrealizedPL', 0)),
                    # SL/TP non sono direttamente qui, ma sono necessari per la chiusura interna del bot
                    'sl': 0.0, 
                    'tp': 0.0,
                })
            
            if float(p.get('short', {}).get('units', 0)) > 0:
                side = 'SELL'
                units = float(p['short']['units'])
                price = float(p['short']['averagePrice'])
                trade_id = f"POS-{instrument}-SHORT"

                formatted_positions.append({
                    'symbol': instrument,
                    'side': side,
                    'qty': units,
                    'entry_price': price,
                    'order_id': trade_id,
                    'pnl': float(p.get('unrealizedPL', 0)),
                    'sl': 0.0,
                    'tp': 0.0,
                })

        return formatted_positions

    except oandapyV20.exceptions.V20Error as e:
        logging.error(f"❌ Errore OANDA V20 API (posizioni): {e}")
        return []
    except Exception as e:
        logging.error(f'❌ Errore nel recupero posizioni OANDA: {e}')
        return []

# Rinomina la funzione per chiarezza, il contenuto è invariato nella logica
async def sync_positions_with_oanda(chat_id, context):
    """Sincronizza le posizioni attive nel bot con quelle reali su OANDA."""
    # ... (Il corpo di questa funzione usa solo get_open_positions_from_oanda) ...
    
    # Usa la nuova funzione
    real_positions = await get_open_positions_from_oanda()
    # ... (la logica di sincronizzazione rimane la stessa) ...
    # ...
    # return True/False (rimane)

# ----------------------------- TELEGRAM HANDLERS -----------------------------

# --- Riscritto per OANDA (Aggiornamento di run_analysis) ---
async def run_analysis(symbol, timeframe, chat_id, autotrade, context):
    # ...
    # riga 387: df = oanda_get_klines(symbol, timeframe)
    df = oanda_get_klines(symbol, timeframe) 
    # ...
    # riga 461: result = place_oanda_order(symbol, side, qty, entry_price, sl_price, tp_price)
    result = place_oanda_order(symbol, side, qty, entry_price, sl_price, tp_price)
    # ...
    # riga 497: await send_telegram_error_message(f"❌ Errore di piazzamento ordine OANDA per {symbol} {timeframe}:\n{result['msg']}", context, chat_id)
    # ...
    # riga 544: await send_telegram_error_message(oanda_format_api_error(error_code, error_msg), context, chat_id)
    # ... (tutti i riferimenti a bybit_get_klines devono usare oanda_get_klines)


# --- Riscritto per OANDA ---
async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /balance - Mostra il saldo dell'account OANDA"""
    # Verifica l'esistenza di API keys e account ID
    if not OANDA_ACCESS_TOKEN or not OANDA_ACCOUNT_ID:
        await update.message.reply_text('⚠️ Configurazione OANDA mancante. Imposta OANDA_ACCESS_TOKEN e OANDA_ACCOUNT_ID.')
        return
    
    try:
        # 1. Ottieni il riepilogo dell'account
        api = create_oanda_api()
        r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
        response = api.request(r)
        
        # Estrai i dati
        account_data = response.get('account', {})
        balance = float(account_data.get('balance', 0.0))
        unrealized_pl = float(account_data.get('unrealizedPL', 0.0))
        margin_used = float(account_data.get('marginUsed', 0.0))
        available_margin = float(account_data.get('marginAvailable', 0.0))
        
        # 2. Formatta il messaggio
        msg = f"🏦 <b>Saldo Account OANDA ({TRADING_MODE.upper()})</b>\n\n"
        msg += f"🆔 Account ID: <code>{OANDA_ACCOUNT_ID}</code>\n"
        msg += f"💵 Saldo Totale: <b>{balance:.2f} {account_data.get('currency', 'USD')}</b>\n"
        msg += f"📈 PnL Non Realizzato: <code>{unrealized_pl:+.2f}</code>\n"
        msg += f"🛡️ Margine Utilizzato: <code>{margin_used:.2f}</code>\n"
        msg += f"💰 Margine Disponibile: <code>{available_margin:.2f}</code>"
        
        await update.message.reply_text(msg, parse_mode='HTML')

    except oandapyV20.exceptions.V20Error as e:
        error_msg = str(e)
        error_code = e.response.status_code if hasattr(e.response, 'status_code') else 0
        formatted_error = oanda_format_api_error(error_code, error_msg)
        await update.message.reply_text(formatted_error, parse_mode='HTML')
    except Exception as e:
        logging.error(f'Errore cmd_balance OANDA: {e}')
        await update.message.reply_text(f'⚠️ Errore generico nel recupero del saldo OANDA: {e}', parse_mode='HTML')


# --- Riscritto per OANDA ---
async def cmd_posizioni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /posizioni - Mostra le posizioni aperte su OANDA e quelle tracciate dal bot."""
    # ... (Logica di posizioni attive dal bot invariata) ...
    
    # 2. Ottieni posizioni reali da OANDA
    real_positions = await get_open_positions_from_oanda()
    
    # ... (Logica di visualizzazione posizioni invariata, usa i dati formattati da get_open_positions_from_oanda) ...
    # ...


# --- Riscritto per OANDA ---
async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /sync - Sincronizza le posizioni tracciate dal bot con quelle reali su OANDA."""
    chat_id = update.effective_chat.id
    await update.message.reply_text('🔄 Sincronizzazione posizioni OANDA in corso...')
    
    # Usa la nuova funzione
    success = await sync_positions_with_oanda(chat_id, context)
    
    if success:
        await update.message.reply_text('✅ Sincronizzazione completata.')
    else:
        await update.message.reply_text('⚠️ Errore nella sincronizzazione con OANDA.\nVerifica le API keys e riprova.')
        
        
# --- Riscritto per OANDA ---
async def cmd_chiudi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /chiudi SYMBOL - Tenta di chiudere la posizione su OANDA e la rimuove dal tracking del bot."""
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) != 1:
        await update.message.reply_text('❌ Uso: /chiudi <SYMBOL> (es. /chiudi EUR_USD)')
        return

    symbol = args[0].upper()

    with POSITIONS_LOCK:
        position_data = ACTIVE_POSITIONS.get(symbol)
        
    if not position_data:
        await update.message.reply_text(f'⚠️ Il bot non sta tracciando una posizione per {symbol}.')
        # Tenta comunque di chiudere su OANDA se una posizione esiste
        pass

    await update.message.reply_text(f'⏳ Tentativo di chiusura della posizione {symbol} su OANDA...')

    # Chiama la funzione di chiusura OANDA (che cerca di chiudere la posizione a mercato)
    # L'endpoint PositionClose chiude tutte le posizioni (long e short) per un dato strumento.
    close_result = await close_oanda_trade(symbol, trade_id='ALL') 

    if close_result['success']:
        with POSITIONS_LOCK:
            if symbol in ACTIVE_POSITIONS:
                del ACTIVE_POSITIONS[symbol]
                
        await update.message.reply_text(f'✅ Posizione {symbol} chiusa con successo su OANDA e rimossa dal tracking del bot.')
        logging.info(f'Posizione {symbol} chiusa e rimossa dal tracking.')
    else:
        await update.message.reply_text(f"❌ Errore nella chiusura della posizione {symbol} su OANDA:\n{close_result['msg']}", parse_mode='HTML')


# --- Riscritto per OANDA (cmd_test) ---
async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /test SYMBOL TF - Esegue un test di pattern detection senza trading."""
    # ...
    # riga 678: df = oanda_get_klines(symbol, timeframe, limit=20)
    df = oanda_get_klines(symbol, timeframe, count=20) 
    # ...

# ----------------------------- MAIN -----------------------------

def main():
    """Setup e avvio del bot."""
    # ... (Logica di inizializzazione invariata) ...
    
    # Aggiungi gli handler dei comandi Telegram
    application.add_handler(CommandHandler('start', cmd_start))
    application.add_handler(CommandHandler('help', cmd_start))
    application.add_handler(CommandHandler('analizza', cmd_analizza))
    application.add_handler(CommandHandler('stop', cmd_stop))
    application.add_handler(CommandHandler('list', cmd_list))
    application.add_handler(CommandHandler('test', cmd_test))
    application.add_handler(CommandHandler('balance', cmd_balance))
    application.add_handler(CommandHandler('pausa', cmd_pausa))
    application.add_handler(CommandHandler('riprendi', cmd_riprendi))
    application.add_handler(CommandHandler('posizioni', cmd_posizioni))
    application.add_handler(CommandHandler('chiudi', cmd_chiudi))
    application.add_handler(CommandHandler('sync', cmd_sync))
    
    # Avvia bot
    mode_emoji = "🎮" if TRADING_MODE == 'practice' else "⚠️💰"
    logging.info('🚀 Bot avviato correttamente!')
    logging.info(f'{mode_emoji} Modalità Trading: {TRADING_MODE.upper()} (OANDA)')
    logging.info(f'⏱️ Timeframes supportati: {ENABLED_TFS}')
    logging.info(f'💰 Rischio per trade: ${RISK_USD}')
    
    if TRADING_MODE == 'live':
        logging.warning('⚠️⚠️⚠️ ATTENZIONE: MODALITÀ LIVE - TRADING REALE! ⚠️⚠️⚠️')
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    except telegram.error.Conflict as e:
        logging.error("❌ ERRORE: Un'altra istanza del bot è in esecuzione.")
    except Exception as e:
        logging.critical(f"🛑 ERRORE CRITICO: Il bot si è bloccato: {e}")


if __name__ == '__main__':
    # Log level settato a INFO per default
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    if not TELEGRAM_TOKEN:
        logging.critical("❌ ERRORE: TELEGRAM_TOKEN non configurato. Il bot non può partire.")
    elif not OANDA_ACCESS_TOKEN or not OANDA_ACCOUNT_ID:
        logging.critical("❌ ERRORE: OANDA_ACCESS_TOKEN o OANDA_ACCOUNT_ID non configurati. Il bot non può fare trading.")
    else:
        main()
