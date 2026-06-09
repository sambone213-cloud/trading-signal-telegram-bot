"""
telegram_notify.py
──────────────────
Lightweight Telegram notifier for QuantDesk.
No third-party library required — uses plain requests to the Bot API.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → get your BOT_TOKEN
  2. Message your new bot once, then run:
       python -c "import telegram_notify; telegram_notify.get_chat_id()"
     Copy the chat_id it prints.
  3. Set environment variables:
       TELEGRAM_BOT_TOKEN=your_token
       TELEGRAM_CHAT_ID=your_chat_id

Usage:
  from telegram_notify import TelegramNotifier
  notifier = TelegramNotifier()
  notifier.send_trade("BUY", "AAPL", 10, 192.50, "EMA Cross")
"""

import os
import time
import requests
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chat_id() -> None:
    """Print the chat_id of the most recent message sent to your bot.
    Run once after you've messaged the bot to discover your chat_id."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN first.")
        return
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    updates = r.json().get("result", [])
    if not updates:
        print("No messages found. Send a message to your bot first, then re-run.")
        return
    for u in updates:
        msg = u.get("message") or u.get("channel_post", {})
        chat = msg.get("chat", {})
        print(f"chat_id: {chat.get('id')}  |  type: {chat.get('type')}  |  name: {chat.get('first_name') or chat.get('title')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Thread-safe Telegram notification client.
    All send_* methods are no-ops when credentials are not configured,
    so it's always safe to call them — no try/except needed at call sites.
    """

    _API_BASE = "https://api.telegram.org/bot{token}/{method}"
    _THROTTLE_SECS = 2  # minimum gap between messages to avoid rate-limit

    def __init__(self,
                 token: str = None,
                 chat_id: str = None):
        self._token   = token   or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._last_sent = 0.0
        self._enabled = bool(self._token and self._chat_id)
        if not self._enabled:
            print("[TelegramNotifier] Credentials not set — notifications disabled.")

    # ── Core sender ───────────────────────────────────────────────────────────

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message. Returns True on success, False otherwise."""
        if not self._enabled:
            return False
        # Simple throttle so we don't hit Telegram's 30-msg/sec limit
        gap = time.time() - self._last_sent
        if gap < self._THROTTLE_SECS:
            time.sleep(self._THROTTLE_SECS - gap)
        url = self._API_BASE.format(token=self._token, method="sendMessage")
        try:
            r = requests.post(url, json={
                "chat_id":    self._chat_id,
                "text":       text,
                "parse_mode": parse_mode,
            }, timeout=8)
            self._last_sent = time.time()
            return r.ok
        except Exception as e:
            print(f"[TelegramNotifier] send error: {e}")
            return False

    # ── Formatted messages ────────────────────────────────────────────────────

    def send_trade(
        self,
        side: str,          # "BUY" or "SELL"
        symbol: str,
        qty: float,
        price: float,
        strategy: str = "",
        pnl: float = None,  # realised P&L on close (for SELL)
        note: str = "",
    ) -> bool:
        arrow = "🟢" if side.upper() == "BUY" else "🔴"
        ts    = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        pnl_line = ""
        if pnl is not None:
            sign  = "+" if pnl >= 0 else ""
            color = "✅" if pnl >= 0 else "⚠️"
            pnl_line = f"\n{color} <b>Realised P&L:</b> {sign}${pnl:,.2f}"
        strat_line = f"\n📐 <b>Strategy:</b> {strategy}" if strategy else ""
        note_line  = f"\n📝 {note}" if note else ""
        text = (
            f"{arrow} <b>TRADE — {side.upper()} {symbol}</b>\n"
            f"──────────────────────\n"
            f"📦 <b>Qty:</b> {qty:,}\n"
            f"💲 <b>Price:</b> ${price:,.4f}"
            f"{strat_line}"
            f"{pnl_line}"
            f"{note_line}\n"
            f"🕐 {ts}"
        )
        return self._send(text)

    def send_pnl_snapshot(
        self,
        equity: float,
        cash: float,
        unrealised_pnl: float,
        realised_pnl: float,
        positions: list[dict] = None,   # [{"symbol": str, "qty": int, "pnl": float}]
        label: str = "DAILY SNAPSHOT",
    ) -> bool:
        ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        upnl_sign = "+" if unrealised_pnl >= 0 else ""
        rpnl_sign = "+" if realised_pnl   >= 0 else ""
        upnl_e = "✅" if unrealised_pnl >= 0 else "🔴"
        rpnl_e = "✅" if realised_pnl   >= 0 else "🔴"

        pos_lines = ""
        if positions:
            pos_lines = "\n\n<b>Open Positions:</b>"
            for p in positions[:8]:  # cap at 8 to keep message readable
                sym  = p.get("symbol", "")
                qty  = p.get("qty", 0)
                ppnl = p.get("pnl", 0.0)
                sign = "+" if ppnl >= 0 else ""
                icon = "▲" if ppnl >= 0 else "▼"
                pos_lines += f"\n  {icon} {sym}  {qty:,} sh  {sign}${ppnl:,.2f}"

        text = (
            f"📊 <b>QuantDesk — {label}</b>\n"
            f"──────────────────────\n"
            f"💼 <b>Equity:</b>  ${equity:>12,.2f}\n"
            f"💵 <b>Cash:</b>    ${cash:>12,.2f}\n"
            f"{upnl_e} <b>Unrealised:</b> {upnl_sign}${unrealised_pnl:,.2f}\n"
            f"{rpnl_e} <b>Realised:</b>   {rpnl_sign}${realised_pnl:,.2f}"
            f"{pos_lines}\n"
            f"──────────────────────\n"
            f"🕐 {ts}"
        )
        return self._send(text)

    def send_signal(
        self,
        symbol: str,
        strategy: str,
        signal: str,        # "BUY" | "SELL" | "HOLD" | "STRONG_BUY" etc.
        price: float,
        details: str = "",  # extra context e.g. "EMA9 crossed above EMA21"
    ) -> bool:
        icon_map = {
            "BUY":        "📈",
            "STRONG_BUY": "🚀",
            "SELL":       "📉",
            "STRONG_SELL":"💥",
            "HOLD":       "⏸",
        }
        icon = icon_map.get(signal.upper(), "🔔")
        ts   = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        det  = f"\n💬 {details}" if details else ""
        text = (
            f"{icon} <b>SIGNAL — {symbol}</b>\n"
            f"──────────────────────\n"
            f"📐 <b>Strategy:</b> {strategy}\n"
            f"⚡ <b>Signal:</b>   {signal.upper()}\n"
            f"💲 <b>Price:</b>    ${price:,.4f}"
            f"{det}\n"
            f"🕐 {ts}"
        )
        return self._send(text)

    def send_alert(
        self,
        symbol: str,
        direction: str,     # "above" | "below"
        threshold: float,
        current_price: float,
    ) -> bool:
        icon = "🔺" if direction == "above" else "🔻"
        ts   = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        text = (
            f"{icon} <b>PRICE ALERT — {symbol}</b>\n"
            f"──────────────────────\n"
            f"Crossed {direction} <b>${threshold:,.2f}</b>\n"
            f"💲 <b>Current:</b> ${current_price:,.4f}\n"
            f"🕐 {ts}"
        )
        return self._send(text)

    def send_raw(self, text: str) -> bool:
        """Send a plain HTML message."""
        return self._send(text)

    def test(self) -> bool:
        """Send a test ping to verify credentials work."""
        return self._send(
            "✅ <b>QuantDesk connected!</b>\n"
            "Telegram notifications are working.\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — pages import this so only one instance is created
# ─────────────────────────────────────────────────────────────────────────────

_notifier: TelegramNotifier | None = None

def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
