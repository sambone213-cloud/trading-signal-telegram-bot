"""
Trading Alert Agent
-------------------
Monitors live market data and fires alerts based on:
  1. QuantDesk strategies (EMA Cross, MACD, BB Reversal, VWAP Reclaim, ORB)
  2. Vishy's rules (RSI + EMA9/21, VIX filter, level proximity, timing, 0DTE sizing)

Alerts print to console AND send via Telegram (if configured).
Does NOT place trades — you pull the trigger.

Usage:
    python trading_alert_agent.py
    python trading_alert_agent.py --symbols SPY QQQ TSLA --interval 30
"""

import argparse
import os
import sys
import time
import datetime
import warnings
import functools
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*utcnow.*")
from typing import Optional

import pandas as pd
import numpy as np

# ── QuantDesk imports ─────────────────────────────────────────────────────────
try:
    from alpaca_client import AlpacaClient
except ImportError:
    AlpacaClient = None


def _bars_to_df(bars: list) -> pd.DataFrame:
    """Convert get_bars_ohlcv() list-of-dicts to a DataFrame with datetime column."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df.rename(columns={"t": "datetime", "o": "open", "h": "high",
                        "l": "low", "c": "close", "v": "volume", "vw": "vwap"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

from trade_strategies import run_all_strategies, StrategySignal

try:
    from telegram_notify import get_notifier as _get_tg
except ImportError:
    def _get_tg():
        class _Noop:
            def send_signal(self, *a, **kw): pass
            def send_alert(self, *a, **kw): pass
        return _Noop()


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS   = ["SPY", "QQQ"]
POLL_INTERVAL_SEC = 15          # seconds between scans
VIX_CAUTION       = 25.0        # above this: warn
VIX_DANGER        = 35.0        # above this: skip new entries
MAX_TRADES_DAY    = 2           # Vishy + Marco rule
OPTION_PREMIUM_LO = 0.30        # target premium range low
OPTION_PREMIUM_HI = 0.60        # target premium range high
STOP_LOSS_PCT     = 0.50        # hard 50% stop per trade


def _fetch_yf_bars(symbol: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    """Fetch bars via yfinance — ~15s delay, matches Robinhood prices closely."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if hist.empty:
            return pd.DataFrame()
        hist.index = hist.index.tz_convert("UTC")
        df = pd.DataFrame({
            "datetime": hist.index,
            "open":     hist["Open"].values,
            "high":     hist["High"].values,
            "low":      hist["Low"].values,
            "close":    hist["Close"].values,
            "volume":   hist["Volume"].values,
        })
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        print(f"  [yf] {symbol}: {e}")
        return pd.DataFrame()


# ── Market session helpers ────────────────────────────────────────────────────

def _et_now() -> datetime.datetime:
    """Current time in US/Eastern as a naive datetime."""
    utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return utc - datetime.timedelta(hours=4)


def _market_open() -> bool:
    now = _et_now()
    if now.weekday() >= 5:        # Saturday / Sunday
        return False
    return datetime.time(9, 30) <= now.time() <= datetime.time(16, 0)


def _in_avoid_window() -> bool:
    """First 15 minutes after open — Vishy skips, Marco skips 30 min."""
    now = _et_now()
    return now.time() < datetime.time(9, 45)


def _minutes_since_open() -> int:
    now = _et_now()
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((now - open_time).total_seconds() / 60))


# ── Indicator calculation ─────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators needed by QuantDesk strategies + Vishy checks."""
    df = df.copy()
    close = df["close"]

    df["ema9"]   = close.ewm(span=9,  adjust=False).mean()
    df["ema21"]  = close.ewm(span=21, adjust=False).mean()
    df["ema50"]  = close.ewm(span=50, adjust=False).mean()

    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2σ)
    sma20         = close.rolling(20).mean()
    std20         = close.rolling(20).std()
    df["bb_mid"]   = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # ATR-14
    high, low = df["high"], df["low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # VWAP (session-level — resets each day via cumulative)
    if "volume" in df.columns:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


# ── VIX fetch ─────────────────────────────────────────────────────────────────

def _fetch_vix(client) -> Optional[float]:
    """Fetch actual VIX index via yfinance (^VIX)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ── Pre-market level calculator ───────────────────────────────────────────────

def calculate_levels(client, symbol: str) -> dict:
    """
    Auto-calculate key levels Vishy marks pre-market:
      - 4hr swing highs/lows (last 10 days daily bars as proxy)
      - 15-min recent swing highs/lows (last 5 sessions)
      - Prior day high/low/close
      - Pre-market high/low (today's bars before 9:30 ET)
    Returns dict with all levels for display + proximity checks.
    """
    levels = {
        "prior_high": None, "prior_low": None, "prior_close": None,
        "swing_highs": [], "swing_lows": [],
        "premarket_high": None, "premarket_low": None,
        "orb_high": None, "orb_low": None,   # filled live after 10:00 ET
    }

    try:
        # Daily bars — prior day H/L/C + multi-day swing levels
        daily = _bars_to_df(client.get_bars_ohlcv(symbol, timeframe="1Day", limit=15)) if client else pd.DataFrame()
        if not daily.empty and len(daily) >= 2:
            prev = daily.iloc[-2]
            levels["prior_high"]  = float(prev["high"])
            levels["prior_low"]   = float(prev["low"])
            levels["prior_close"] = float(prev["close"])

            # Simple swing detection on daily: local highs/lows over 15 bars
            highs = daily["high"].values
            lows  = daily["low"].values
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    levels["swing_highs"].append(round(float(highs[i]), 2))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    levels["swing_lows"].append(round(float(lows[i]), 2))

        # 15-min bars for intraday swing highs/lows (last 3 sessions ≈ 78 bars)
        bars_15 = _bars_to_df(client.get_bars_ohlcv(symbol, timeframe="15Min", limit=78)) if client else pd.DataFrame()
        if not bars_15.empty and len(bars_15) >= 5:
            h15 = bars_15["high"].values
            l15 = bars_15["low"].values
            for i in range(2, len(h15) - 2):
                if h15[i] == max(h15[i-2:i+3]):
                    levels["swing_highs"].append(round(float(h15[i]), 2))
                if l15[i] == min(l15[i-2:i+3]):
                    levels["swing_lows"].append(round(float(l15[i]), 2))

        # Pre-market bars (1-min bars today before 13:30 UTC = 9:30 ET)
        bars_1m = _bars_to_df(client.get_bars_ohlcv(symbol, timeframe="1Min", limit=120)) if client else pd.DataFrame()
        if not bars_1m.empty:
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            if "datetime" in bars_1m.columns:
                premarket = bars_1m[
                    (bars_1m["datetime"].dt.strftime("%Y-%m-%d") == today_str) &
                    (bars_1m["datetime"].dt.hour < 13) |
                    ((bars_1m["datetime"].dt.hour == 13) &
                     (bars_1m["datetime"].dt.minute < 30))
                ]
                if not premarket.empty:
                    levels["premarket_high"] = round(float(premarket["high"].max()), 2)
                    levels["premarket_low"]  = round(float(premarket["low"].min()), 2)

        # Deduplicate swing levels — keep unique within $0.50
        def _dedup(vals):
            vals = sorted(set(vals), reverse=True)
            out = []
            for v in vals:
                if not out or abs(v - out[-1]) > 0.50:
                    out.append(v)
            return out

        levels["swing_highs"] = _dedup(levels["swing_highs"])[:5]
        levels["swing_lows"]  = _dedup(levels["swing_lows"])[:5]

    except Exception as e:
        print(f"  [{symbol}] Level calc error: {e}")

    return levels


def print_premarket_briefing(client, symbols: list):
    """Print the full pre-market level map — run this before market open."""
    print(f"\n{'='*60}")
    print(f"  PRE-MARKET BRIEFING  {datetime.date.today()}  (Draw these in Robinhood Legend)")
    print(f"{'='*60}")

    all_levels = {}
    for symbol in symbols:
        print(f"\n  {symbol}")
        lvls = calculate_levels(client, symbol)
        all_levels[symbol] = lvls

        if lvls["prior_close"]:
            print(f"    Prior Close : ${lvls['prior_close']:.2f}")
        if lvls["prior_high"] and lvls["prior_low"]:
            print(f"    Prior H/L   : ${lvls['prior_high']:.2f} / ${lvls['prior_low']:.2f}")
        if lvls["premarket_high"] and lvls["premarket_low"]:
            print(f"    Pre-Mkt H/L : ${lvls['premarket_high']:.2f} / ${lvls['premarket_low']:.2f}  ← first candle levels")
        if lvls["swing_highs"]:
            print(f"    Swing Highs : {', '.join(f'${v}' for v in lvls['swing_highs'])}")
        if lvls["swing_lows"]:
            print(f"    Swing Lows  : {', '.join(f'${v}' for v in lvls['swing_lows'])}")

        # Flatten all levels into a single list for the scanner
        flat = []
        for k in ("prior_high", "prior_low", "prior_close", "premarket_high", "premarket_low"):
            if lvls[k]:
                flat.append(lvls[k])
        flat += lvls["swing_highs"] + lvls["swing_lows"]
        all_levels[symbol]["_flat"] = sorted(set(flat))

        print(f"    All levels  : {', '.join(f'${v:.2f}' for v in all_levels[symbol]['_flat'])}")

    print(f"\n{'='*60}\n")
    return all_levels


# ── Level proximity check ─────────────────────────────────────────────────────

def _near_level(price: float, levels: list, atr: float) -> Optional[float]:
    """Return the nearest level if price is within 0.5 × ATR of it."""
    if not levels or atr <= 0:
        return None
    for lvl in levels:
        if abs(price - lvl) <= 0.5 * atr:
            return lvl
    return None


# ── 0DTE option sizing guidance ───────────────────────────────────────────────

def _option_guidance(sig: StrategySignal, underlying_price: float, vix: Optional[float] = None) -> str:
    """
    Translate a StrategySignal into 0DTE option sizing language.
    When VIX is elevated, go deeper OTM to stay in the $0.30-$0.60 premium range.
    VIX 20-25 → start at 4 OTM, VIX 25+ → start at 6 OTM.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    direction = "CALL" if sig.side == "long" else "PUT"

    # Adjust starting OTM offset based on VIX — higher VIX = more expensive premiums
    if vix and vix >= 25:
        start_otm = 6
        vix_note = "  ⚠ VIX elevated — strikes shifted deeper OTM for premium target"
    elif vix and vix >= 20:
        start_otm = 4
        vix_note = "  ℹ VIX elevated — strikes shifted deeper OTM for premium target"
    else:
        start_otm = 1
        vix_note = ""

    tick = 1 if underlying_price > 100 else 0.5
    base = round(underlying_price / tick) * tick
    if sig.side == "long":
        strikes = [base + i * tick for i in range(start_otm, start_otm + 3)]
    else:
        strikes = [base - i * tick for i in range(start_otm, start_otm + 3)]

    strike_str = " / ".join(f"${s:.0f}" for s in strikes[:3])
    otm_label  = f"{start_otm}-{start_otm+2} OTM"

    return (
        f"  0DTE {direction} · Exp {today}\n"
        f"  Strikes ({otm_label}): {strike_str}\n"
        f"  Target premium: ${OPTION_PREMIUM_LO:.2f}–${OPTION_PREMIUM_HI:.2f}\n"
        f"  Stop loss: {STOP_LOSS_PCT*100:.0f}% of premium\n"
        f"  Max 2 trades today · 1 contract ITM max"
        + (f"\n{vix_note}" if vix_note else "")
    )


# ── Alert formatting ──────────────────────────────────────────────────────────

def _format_alert(
    symbol: str,
    sig: StrategySignal,
    underlying_price: float,
    vix: Optional[float],
    near_lvl: Optional[float],
    mins_open: int,
) -> str:
    side_emoji = "🟢 LONG" if sig.side == "long" else "🔴 SHORT"
    reasons    = "  ·  ".join(sig.reasons)
    vix_str    = f"VIX {vix:.1f}" if vix else "VIX N/A"
    lvl_str    = f"Near level ${near_lvl:.2f}" if near_lvl else "No key level nearby"

    return (
        f"\n{'='*56}\n"
        f"  ALERT  {symbol}  {side_emoji}\n"
        f"  Strategy : {sig.strategy}\n"
        f"  Price    : ${underlying_price:.2f}   Confidence: {sig.confidence*100:.0f}%\n"
        f"  Signals  : {reasons}\n"
        f"  {vix_str}   {lvl_str}   {mins_open}m since open\n"
        f"{'─'*56}\n"
        f"{_option_guidance(sig, underlying_price, vix)}\n"
        f"{'='*56}\n"
    )


# ── Trade tracker (enforce max 2/day rule) ───────────────────────────────────

class DailyTracker:
    def __init__(self):
        self._date  = None
        self._count = 0

    def can_trade(self) -> bool:
        today = datetime.date.today()
        if self._date != today:
            self._date  = today
            self._count = 0
        return self._count < MAX_TRADES_DAY

    def record(self):
        self._count += 1

    @property
    def trades_today(self) -> int:
        today = datetime.date.today()
        if self._date != today:
            return 0
        return self._count


# ── Signal deduplication — suppress repeat fires until strategy resets ────────

_last_signal: dict = {}   # symbol -> {strategy -> last_fire_price}

def _is_duplicate(symbol: str, sig: StrategySignal, price: float, atr: float) -> bool:
    """Return True if same strategy fired within 2×ATR of last fire price."""
    key = (symbol, sig.strategy, sig.side)
    last = _last_signal.get(key)
    if last is not None and abs(price - last) < 2 * atr:
        return True
    _last_signal[key] = price
    return False


# ── Main scanner ─────────────────────────────────────────────────────────────

def scan(client, symbols: list, tracker: DailyTracker, key_levels: dict):
    """One full scan pass across all symbols."""

    if not _market_open():
        print(f"[{_et_now().strftime('%H:%M')} ET] Market closed — waiting...")
        return

    if _in_avoid_window():
        print(f"[{_et_now().strftime('%H:%M')} ET] First 15 min — skip (Vishy rule)")
        return

    mins_open = _minutes_since_open()

    if not tracker.can_trade():
        print(f"[{_et_now().strftime('%H:%M')} ET] Max {MAX_TRADES_DAY} trades reached today ({tracker.trades_today}/{MAX_TRADES_DAY}) — monitoring only")

    vix = _fetch_vix(client)
    if vix and vix >= VIX_DANGER:
        print(f"[{_et_now().strftime('%H:%M')} ET] ⚠️  VIX {vix:.1f} >= {VIX_DANGER} — skipping new entries")
        return
    if vix and vix >= VIX_CAUTION:
        print(f"[{_et_now().strftime('%H:%M')} ET] ⚠️  VIX {vix:.1f} — elevated, trade carefully")

    for symbol in symbols:
        try:
            raw = _fetch_yf_bars(symbol, interval="1m", period="1d")
            bars = raw if not raw.empty else (
                _bars_to_df(client.get_bars_ohlcv(symbol, timeframe="1Min", limit=200))
                if client else pd.DataFrame()
            )
            if bars.empty or len(bars) < 20:
                print(f"  [{symbol}] Not enough bars ({len(bars)})")
                continue

            df = _compute_indicators(bars)
            signals = run_all_strategies(df, symbol=symbol, notify=False)

            price    = float(df["close"].iloc[-1])
            atr_val  = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0
            rsi_val  = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
            ema9_val = float(df["ema9"].iloc[-1]) if "ema9" in df.columns else 0
            ema21_val= float(df["ema21"].iloc[-1]) if "ema21" in df.columns else 0

            # Vishy: print current indicator snapshot each scan
            ema_trend = "EMA9>21 ↑" if ema9_val > ema21_val else "EMA9<21 ↓"
            vix_str   = f"VIX {vix:.1f}" if vix else "VIX N/A"
            print(
                f"  [{symbol}] ${price:.2f}  RSI {rsi_val:.0f}  {ema_trend}  "
                f"ATR {atr_val:.2f}  {vix_str}"
            )

            if not signals:
                continue

            levels = key_levels.get(symbol, [])

            for sig in signals:
                if _is_duplicate(symbol, sig, price, atr_val):
                    continue

                # BB Reversal requires 10:30 ET minimum — mean reversion needs time to develop
                if sig.strategy == "BB Reversal" and _et_now().time() < datetime.time(10, 30):
                    continue

                near_lvl = _near_level(price, levels, atr_val)

                alert_text = _format_alert(
                    symbol, sig, price, vix, near_lvl, mins_open
                )
                print(alert_text)

                # Telegram
                try:
                    tg_signal = "BUY" if sig.side == "long" else "SELL"
                    details   = "  ·  ".join(sig.reasons)
                    _get_tg().send_signal(
                        symbol=symbol,
                        strategy=sig.strategy,
                        signal=tg_signal,
                        price=price,
                        details=details + f"\n{_option_guidance(sig, price, vix)}",
                    )
                except Exception:
                    pass

                # Trade count is managed manually — type "traded" in terminal to log

        except Exception as e:
            print(f"  [{symbol}] Error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuantDesk Trading Alert Agent")
    parser.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", type=int,   default=POLL_INTERVAL_SEC,
                        help="Seconds between scans")
    parser.add_argument("--levels",   nargs="*",  default=[],
                        help="Optional extra levels to add (e.g. 540.0 542.5). Auto-levels always run.")
    args = parser.parse_args()

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    # Alpaca is optional — yfinance is the primary data source
    client = None
    if AlpacaClient and api_key and api_secret:
        try:
            client = AlpacaClient(api_key=api_key, api_secret=api_secret)
            print("  Data     : yfinance (primary) + Alpaca (fallback)")
        except Exception:
            print("  Data     : yfinance only (Alpaca init failed)")
    else:
        print("  Data     : yfinance only")
    tracker = DailyTracker()

    print(f"\n{'='*56}")
    print(f"  QuantDesk Alert Agent")
    print(f"  Symbols  : {', '.join(args.symbols)}")
    print(f"  Interval : {args.interval}s")
    print(f"  Rules    : skip first 15m · VIX filter · max {MAX_TRADES_DAY}/day")
    print(f"  Target   : 0DTE $0.30–$0.60 premium · 1-3 OTM · 50% SL")
    print(f"{'='*56}\n")

    # Auto-calculate pre-market levels for all symbols
    all_level_data = print_premarket_briefing(client, args.symbols)

    # Merge auto-levels with any manually passed levels
    key_levels = {}
    for s in args.symbols:
        auto = all_level_data.get(s, {}).get("_flat", [])
        manual = [float(l) for l in args.levels]
        key_levels[s] = sorted(set(auto + manual))

    print("Levels loaded. Starting scanner — open Robinhood Legend and draw the levels above.\n")

    _last_heartbeat = 0.0
    HEARTBEAT_INTERVAL = 600  # send Telegram status every 10 minutes

    while True:
        now_str = _et_now().strftime("%H:%M:%S ET")
        print(f"\n[{now_str}] Scanning {', '.join(args.symbols)}...")
        scan(client, args.symbols, tracker, key_levels)

        # Telegram heartbeat every 10 min so you know it's alive on your phone
        if time.time() - _last_heartbeat >= HEARTBEAT_INTERVAL:
            try:
                vix_now = _fetch_vix() or 0.0
                # get latest price/RSI from last scan for each symbol
                status_lines = []
                for sym in args.symbols:
                    try:
                        raw = _fetch_yf_bars(sym, interval="1m", period="1d")
                        if not raw.empty:
                            df_h = _compute_indicators(raw)
                            p    = float(df_h["close"].iloc[-1])
                            rsi  = float(df_h["rsi"].iloc[-1])
                            trend = "↑" if float(df_h["ema9"].iloc[-1]) > float(df_h["ema21"].iloc[-1]) else "↓"
                            status_lines.append(f"{sym} ${p:.2f}  RSI {rsi:.0f}  {trend}")
                    except Exception:
                        pass
                body = "\n".join(status_lines) or "scanning..."
                _get_tg().send_raw(
                    f"🤖 <b>QuantDesk Heartbeat</b>\n"
                    f"──────────────────────\n"
                    f"{body}\n"
                    f"VIX {vix_now:.1f}  |  {now_str}\n"
                    f"No signals yet — watching 👀"
                )
                _last_heartbeat = time.time()
            except Exception:
                pass

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
