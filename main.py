from telethon import TelegramClient, events
from datetime import datetime
import json
import os
import re

# ================== CONFIG ==================
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = "tg_session"

SIGNAL_PATH = "signal.json"

CHANNELS = [
    "nomecanale1",
    "nomecanale2"
]

# ============================================

# REGEX ANTI-FAKE
PAIR_REGEX = re.compile(r"\bXAUUSD\b", re.IGNORECASE)
SIDE_REGEX = re.compile(r"\b(BUY|SELL)\b", re.IGNORECASE)
ENTRY_REGEX = re.compile(
    r"ENTRY\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE
)

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

@client.on(events.NewMessage(chats=CHANNELS))
async def handler(event):

    # ❌ ignora edit e forward
    if event.message.edit_date or event.message.fwd_from:
        return

    text = event.raw_text.upper()

    # ❌ validazione struttura
    if not PAIR_REGEX.search(text):
        return

    side_match = SIDE_REGEX.search(text)
    entry_match = ENTRY_REGEX.search(text)

    if not side_match or not entry_match:
        return

    side = side_match.group(1)
    entry_low = float(entry_match.group(1))
    entry_high = float(entry_match.group(2))

    # sicurezza extra
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    signal = {
        "symbol": "XAUUSD",
        "action": side,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(SIGNAL_PATH, "w") as f:
        json.dump(signal, f)

    print(f"[OK] {side} XAUUSD {entry_low}-{entry_high}")

async def main():
    await client.start()
    print("🤖 Telegram signal bot avviato")
    await client.run_until_disconnected()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
