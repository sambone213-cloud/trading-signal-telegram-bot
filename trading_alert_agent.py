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

from trade_strategies import run_all_strategies, StrategySignal, check_exit_conditions
from position_manager import PositionManager
from exit_manager import ExitManager
from level_trader import LevelMonitor, briefing_plays

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
        df = df.sort_values("datetime").reset_index(drop=True)
        # Drop the last bar if it has zero volume — it's an in-progress bar.
        # All volume-gated strategies divide by this bar's volume; a 0 gives vol 0.0x
        # which permanently blocks Momentum Flip, VWAP Reclaim, Trend Breakout, ORB.
        if len(df) > 1 and df["volume"].iloc[-1] == 0:
            df = df.iloc[:-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [yf] {symbol}: {e}")
        return pd.DataFrame()


# ── Market session helpers ────────────────────────────────────────────────────

try:
    from zoneinfo import ZoneInfo
    _ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    _ET_ZONE = None   # fall back to fixed offset below


def _et_now() -> datetime.datetime:
    """Current time in US/Eastern as a naive datetime. DST-aware via zoneinfo."""
    utc = datetime.datetime.now(datetime.timezone.utc)
    if _ET_ZONE is not None:
        return utc.astimezone(_ET_ZONE).replace(tzinfo=None)
    return utc.replace(tzinfo=None) - datetime.timedelta(hours=4)


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

    # Keltner Channels (EMA20 mid ± 2×ATR10) — used by Keltner Bounce strategy
    ema20_kc = close.ewm(span=20, adjust=False).mean()
    atr10    = tr.rolling(10).mean()
    df["kc_mid"]   = ema20_kc
    df["kc_upper"] = ema20_kc + 2 * atr10
    df["kc_lower"] = ema20_kc - 2 * atr10

    # VWAP (session-level — resets each trading day, not cumulative across days)
    if "volume" in df.columns:
        if "datetime" in df.columns:
            day = df["datetime"].dt.date
            df["vwap"] = ((df["close"] * df["volume"]).groupby(day).cumsum()
                          / df["volume"].groupby(day).cumsum())
        else:
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # ADX-14
    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    overlap  = (df["high"].diff() > 0) & (-df["low"].diff() > 0)
    plus_dm[overlap & (df["high"].diff() < -df["low"].diff())]  = 0
    minus_dm[overlap & (df["high"].diff() > -df["low"].diff())] = 0
    atr14    = tr.rolling(14).mean()
    df["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    # VWMA-9 and VWMA-21
    if "volume" in df.columns:
        df["vwma9"]    = (close * df["volume"]).rolling(9).sum()  / df["volume"].rolling(9).sum()
        df["vwma21"]   = (close * df["volume"]).rolling(21).sum() / df["volume"].rolling(21).sum()
        vol_ma20       = df["volume"].rolling(20, min_periods=5).mean()
        df["vol_ratio"] = df["volume"] / vol_ma20.replace(0, np.nan)

    # Momentum / time helpers — per-bar from datetime column (needed for ORB)
    df["mom10"] = close - close.shift(10)
    if "datetime" in df.columns:
        if _ET_ZONE is not None:
            dt_et = df["datetime"].dt.tz_convert(_ET_ZONE)
            df["hour_et"]   = dt_et.dt.hour
            df["minute_et"] = dt_et.dt.minute
        else:
            dt_et = df["datetime"] - pd.Timedelta(hours=4)
            df["hour_et"]   = dt_et.dt.hour
            df["minute_et"] = dt_et.dt.minute
        df["mins_since_open"] = ((df["hour_et"] - 9) * 60 + df["minute_et"] - 30).clip(lower=0)
    else:
        now_et = _et_now()
        df["hour_et"]         = now_et.hour
        df["minute_et"]       = now_et.minute
        df["mins_since_open"] = max(0, (now_et.hour - 9) * 60 + now_et.minute - 30)

    return df


# ── VIX fetch ─────────────────────────────────────────────────────────────────

def _fetch_vix(client=None) -> Optional[float]:
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


# ── Market-open Telegram briefing ─────────────────────────────────────────────

def build_open_briefing(client, symbol: str):
    """
    Build the market-open briefing text for one symbol.
    Returns (text, flat_levels) — flat_levels refreshes the scanner's key
    levels for the day. Uses Alpaca levels when available, yfinance fallback
    for prior-day OHLC, premarket H/L, gap, and daily ATR.
    """
    lvls = calculate_levels(client, symbol)
    today_et = _et_now().date()
    daily_atr = None
    gap_pct = None
    pre_last = None

    try:
        import yfinance as yf

        # Prior-day OHLC + daily ATR(14) from daily bars
        daily = yf.Ticker(symbol).history(period="2mo", interval="1d", auto_adjust=False)
        if not daily.empty:
            if daily.index.tz is not None and _ET_ZONE is not None:
                daily.index = daily.index.tz_convert(_ET_ZONE)
            prior = daily[daily.index.date < today_et]
            if len(prior) >= 1 and not lvls.get("prior_close"):
                prev = prior.iloc[-1]
                lvls["prior_close"] = round(float(prev["Close"]), 2)
                lvls["prior_high"]  = round(float(prev["High"]), 2)
                lvls["prior_low"]   = round(float(prev["Low"]), 2)
            if len(prior) >= 15:
                h, l, pc = prior["High"], prior["Low"], prior["Close"].shift(1)
                tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
                daily_atr = float(tr.rolling(14).mean().iloc[-1])

        # Premarket H/L and gap from today's extended-hours 1-min bars
        pre = yf.Ticker(symbol).history(period="1d", interval="1m",
                                        prepost=True, auto_adjust=False)
        if not pre.empty:
            if pre.index.tz is not None and _ET_ZONE is not None:
                pre.index = pre.index.tz_convert(_ET_ZONE)
            pre_today = pre[(pre.index.date == today_et) &
                            (pre.index.time < datetime.time(9, 30))]
            if not pre_today.empty:
                lvls["premarket_high"] = round(float(pre_today["High"].max()), 2)
                lvls["premarket_low"]  = round(float(pre_today["Low"].min()), 2)
                pre_last = float(pre_today["Close"].iloc[-1])
                if lvls.get("prior_close"):
                    gap_pct = (pre_last - lvls["prior_close"]) / lvls["prior_close"] * 100
    except Exception as e:
        print(f"  [briefing] {symbol} data fetch: {e}")

    vix = _fetch_vix()

    # ── Outlook lines (data-driven, no predictions) ──────────────────────────
    outlook = []
    if gap_pct is not None:
        if gap_pct >= 0.3:
            outlook.append(f"Gap UP {gap_pct:+.2f}% — trend-day potential; "
                           f"watch gap-fill to ${lvls['prior_close']:.2f} first")
        elif gap_pct <= -0.3:
            outlook.append(f"Gap DOWN {gap_pct:+.2f}% — trend-day potential; "
                           f"bounce to ${lvls['prior_close']:.2f} would fill the gap")
        else:
            outlook.append(f"Flat open ({gap_pct:+.2f}%) — likely chop early, "
                           f"let ORB define direction by 10:00")
    if daily_atr and lvls.get("prior_close"):
        lo = lvls["prior_close"] - daily_atr
        hi = lvls["prior_close"] + daily_atr
        outlook.append(f"Expected range (ATR14 ${daily_atr:.2f}): ${lo:.2f} – ${hi:.2f}")
    if vix:
        if vix >= VIX_DANGER:
            outlook.append(f"VIX {vix:.1f} — DANGER, bot skips new entries today")
        elif vix >= VIX_CAUTION:
            outlook.append(f"VIX {vix:.1f} — elevated: 6+ OTM strikes for $0.30–0.60, size down")
        elif vix >= 20:
            outlook.append(f"VIX {vix:.1f} — moderately elevated: start 4 OTM")
        else:
            outlook.append(f"VIX {vix:.1f} — calm: 1–3 OTM strikes in premium range")

    # ── Assemble message ─────────────────────────────────────────────────────
    lines = [f"☀️ <b>MARKET OPEN — {symbol}</b>  {today_et.strftime('%a %b %d')}",
             "──────────────────────"]
    if lvls.get("prior_close"):
        lines.append(f"Prior close ${lvls['prior_close']:.2f}  |  "
                     f"H ${lvls['prior_high']:.2f} / L ${lvls['prior_low']:.2f}")
    if lvls.get("premarket_high"):
        pre_str = f"Pre-mkt H ${lvls['premarket_high']:.2f} / L ${lvls['premarket_low']:.2f}"
        if pre_last:
            pre_str += f"  |  last ${pre_last:.2f}"
        lines.append(pre_str)
    if lvls.get("swing_highs"):
        lines.append("Swing highs: " + ", ".join(f"${v}" for v in lvls["swing_highs"]))
    if lvls.get("swing_lows"):
        lines.append("Swing lows: " + ", ".join(f"${v}" for v in lvls["swing_lows"]))
    if outlook:
        lines.append("──────────────────────")
        lines.extend(outlook)

    # Flat level list (also returned for the scanner's proximity + level monitor)
    flat = [lvls[k] for k in ("prior_high", "prior_low", "prior_close",
                              "premarket_high", "premarket_low") if lvls.get(k)]
    flat += lvls.get("swing_highs", []) + lvls.get("swing_lows", [])
    flat = sorted(set(flat))

    # Level plays — break/retest/target around the expected open price
    ref_price = pre_last or lvls.get("prior_close")
    if ref_price and daily_atr:
        # daily_atr is a daily range; use a 1-min-ish fraction for intraday buffers
        plays = briefing_plays(flat, ref_price, max(daily_atr * 0.08, 0.15))
        if plays:
            lines.append("──────────────────────")
            lines.append("📐 Level plays:")
            lines.extend(plays)

    lines.append("──────────────────────")
    lines.append("Today's windows:")
    lines.append("9:45–10:00  Opening Drive (DI direction)")
    lines.append("10:00+      ORB breakout watch")
    lines.append("12:00       Lunch VWAP Hold")
    lines.append("3:00–3:55   Power Hour Dip")
    lines.append("All day     Dip Buy · Keltner · Momentum · Trend · VWAP · BB+ADX")

    return "\n".join(lines), flat


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

_last_signal: dict = {}          # (symbol,strategy,side) -> last_fire_price
_last_signal_time: dict = {}     # (symbol,strategy,side) -> last_fire datetime
_entry_times: dict = {}          # symbol -> entry datetime for bars_held calculation
_signal_day = None               # resets the dedup memory each trading day

SIGNAL_COOLDOWN_MIN = 20         # min minutes before same strategy/side re-alerts

def _is_duplicate(symbol: str, sig: StrategySignal, price: float, atr: float) -> bool:
    """
    Suppress a repeat if the same strategy/side fired EITHER within 2xATR of its
    last fire price OR within the cooldown window. The time gate is what tames
    the over-firers — Lunch VWAP and Opening Drive were alerting 4-5x/day.
    """
    global _signal_day
    now = _et_now()
    today = now.date()
    if _signal_day != today:
        _last_signal.clear()
        _last_signal_time.clear()
        _signal_day = today
    key = (symbol, sig.strategy, sig.side)
    last = _last_signal.get(key)
    last_t = _last_signal_time.get(key)
    if last is not None and abs(price - last) < 2 * atr:
        return True
    if last_t is not None and (now - last_t).total_seconds() < SIGNAL_COOLDOWN_MIN * 60:
        return True
    _last_signal[key] = price
    _last_signal_time[key] = now
    return False


# ── Main scanner ─────────────────────────────────────────────────────────────

def scan(client, symbols: list, tracker: DailyTracker, key_levels: dict, pm: PositionManager = None, em: ExitManager = None, lm: LevelMonitor = None):
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
            # 2 days of bars: prior session warms up EMA50/ADX/DI so strategies
            # can fire at the open — with 1d, the >=35-bar minimum meant nothing
            # could fire before ~10:05 and Opening Drive (9:45-10:00) was dead.
            raw = _fetch_yf_bars(symbol, interval="1m", period="2d")
            bars = raw if not raw.empty else (
                _bars_to_df(client.get_bars_ohlcv(symbol, timeframe="1Min", limit=600))
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

            # Diagnostics — show how close each strategy's gating conditions are
            adx_val   = float(df["adx"].iloc[-1]) if "adx" in df.columns and pd.notna(df["adx"].iloc[-1]) else None
            vr_val    = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns and pd.notna(df["vol_ratio"].iloc[-1]) else None
            bb_lo     = float(df["bb_lower"].iloc[-1]) if "bb_lower" in df.columns else None
            bb_hi     = float(df["bb_upper"].iloc[-1]) if "bb_upper" in df.columns else None
            kc_lo     = float(df["kc_lower"].iloc[-1]) if "kc_lower" in df.columns else None
            kc_hi     = float(df["kc_upper"].iloc[-1]) if "kc_upper" in df.columns else None
            macd_val  = float(df["macd"].iloc[-1]) if "macd" in df.columns else None
            macd_sig  = float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else None

            adx_str = f"ADX {adx_val:.0f}" if adx_val is not None else "ADX n/a"
            vr_str  = f"vol {vr_val:.1f}x" if vr_val is not None else "vol n/a"

            if bb_hi is not None and price > bb_hi:
                bb_pos = "BB:above_upper"
            elif bb_lo is not None and price < bb_lo:
                bb_pos = "BB:below_lower"
            else:
                bb_pos = "BB:inside"

            if kc_hi is not None and price > kc_hi:
                kc_pos = "KC:above_upper"
            elif kc_lo is not None and price < kc_lo:
                kc_pos = "KC:below_lower"
            else:
                kc_pos = "KC:inside"

            macd_pos = "MACD:n/a"
            if macd_val is not None and macd_sig is not None:
                macd_pos = "MACD:bull" if macd_val > macd_sig else "MACD:bear"

            # Vishy: print current indicator snapshot each scan
            ema_trend = "EMA9>21 ↑" if ema9_val > ema21_val else "EMA9<21 ↓"
            vix_str   = f"VIX {vix:.1f}" if vix else "VIX N/A"
            try:
                from trade_strategies import market_regime
                regime_str = market_regime(df)
            except Exception:
                regime_str = "?"
            print(
                f"  [{symbol}] ${price:.2f}  RSI {rsi_val:.0f}  {ema_trend}  "
                f"ATR {atr_val:.2f}  {vix_str}  {adx_str}  {vr_str}  "
                f"{bb_pos}  {kc_pos}  {macd_pos}  REGIME:{regime_str}"
            )

            # Level plays — break / retest / continuation on the day's key levels
            if lm is not None:
                try:
                    for play in lm.update(symbol, key_levels.get(symbol, []),
                                          price, atr_val, _et_now(), regime_str):
                        print(f"\n{play}\n")
                        try:
                            _get_tg().send_raw(play)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"  [level] {symbol}: {e}")

            # Exit manager — evaluate open positions every scan
            if em:
                try:
                    current_prem = float(df["atr"].iloc[-1]) * 0.5
                    exit_reason = em.evaluate(symbol, current_prem, df)
                    # TIER1 = scale out half, RUNNER = still holding — only a full
                    # exit should free the position manager slot
                    if exit_reason and "TIER1" not in exit_reason and "RUNNER" not in exit_reason:
                        if pm:
                            pm.force_close(_et_now())
                except Exception:
                    pass

            # Strategy-level exit conditions — check if open position should be exited
            if pm and pm._active_side and pm._active_strategy:
                try:
                    entry_dt  = _entry_times.get(symbol)
                    bars_held = int(((_et_now() - entry_dt).total_seconds() / 60)) if entry_dt else 999
                    exit_hint = check_exit_conditions(df, pm._active_side, pm._active_strategy, bars_held)
                    if exit_hint:
                        msg = (
                            f"⚠️  EXIT SIGNAL  {symbol}\n"
                            f"──────────────────────\n"
                            f"Position : {pm._active_side.upper()} via {pm._active_strategy}\n"
                            f"Signal   : {exit_hint}\n"
                            f"Price    : ${price:.2f}\n"
                            f"🕐 {_et_now().strftime('%H:%M ET')}"
                        )
                        print(f"\n{msg}")
                        try:
                            _get_tg().send_raw(msg)
                        except Exception:
                            pass
                except Exception:
                    pass

            if not signals:
                continue

            levels = key_levels.get(symbol, [])

            for sig in signals:
                if _is_duplicate(symbol, sig, price, atr_val):
                    continue

                # Mean reversion strategies require 10:30 ET minimum
                if sig.strategy in ("BB+ADX Reversal", "Keltner Bounce") and _et_now().time() < datetime.time(10, 30):
                    continue

                # No gating — every deduped signal alerts in full. Sam keeps his
                # own trade discipline; the bot informs, the human decides.
                # PM just tracks the latest signal as the virtual position so
                # exit alerts stay relevant to the most recent setup.
                if pm:
                    bar_time = _et_now()
                    pm.register(sig, bar_time)
                    _entry_times[symbol] = bar_time

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
                        details=details
                                + f"\nConfidence: {sig.conf_label}"
                                + f"\n{_option_guidance(sig, price, vix)}",
                    )
                except Exception:
                    pass

                # Register entry in exit manager — starts tracking this position
                if em:
                    try:
                        entry_premium = atr_val * 0.5  # rough proxy; replace with actual premium if known
                        em.register_entry(symbol, sig, entry_premium, df)
                    except Exception:
                        pass

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
    # accept both names — .env / Railway use ALPACA_SECRET_KEY
    api_secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY", "")

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
    pm      = PositionManager(lockout_minutes=15, max_trades_per_day=999)  # alerts never gated
    em      = ExitManager(notifier=_get_tg())
    lm      = LevelMonitor()

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

    # One explicit startup ping so deploys/restarts are distinguishable from
    # heartbeats — repeated "restarted" messages = crash loop, investigate.
    try:
        _get_tg().send_raw(
            f"🚀 <b>QuantDesk restarted</b> (deploy or crash recovery)\n"
            f"Market {'OPEN' if _market_open() else 'closed'}  |  "
            f"{_et_now().strftime('%H:%M:%S ET')}\n"
            f"Heartbeats: every 10 min while market open"
        )
    except Exception:
        pass

    _last_heartbeat = time.time()   # first regular heartbeat comes after a full interval
    HEARTBEAT_INTERVAL        = 600    # market hours: every 10 minutes
    HEARTBEAT_CLOSED_INTERVAL = 14400  # market closed: every 4 hours, light ping
    _briefing_day = None               # market-open briefing sent once per trading day
    _report_day   = None               # EOD report card sent once per trading day

    while True:
        now_str = _et_now().strftime("%H:%M:%S ET")
        print(f"\n[{now_str}] Scanning {', '.join(args.symbols)}...")

        # EOD report card at 16:05 ET — replays the day, scores every signal,
        # sends the running per-strategy track record. Window-bounded so an
        # evening redeploy doesn't resend it (run report_card.py manually then).
        now_et = _et_now()
        if (now_et.weekday() < 5
                and datetime.time(16, 5) <= now_et.time() < datetime.time(18, 0)
                and _report_day != now_et.date()):
            _report_day = now_et.date()
            try:
                from report_card import build_report
                for sym in args.symbols:
                    text = build_report(sym)
                    print(f"\n{text}\n")
                    _get_tg().send_raw(text)
            except Exception as e:
                print(f"  [report] {e}")

        # Market-open briefing: levels + outlook, once per day at first scan
        # after 9:30. Also refreshes key levels (boot-time levels go stale).
        if _market_open() and _briefing_day != _et_now().date():
            for sym in args.symbols:
                try:
                    text, flat = build_open_briefing(client, sym)
                    print(f"\n{text}\n")
                    _get_tg().send_raw(text)
                    if flat:
                        manual = [float(l) for l in args.levels]
                        key_levels[sym] = sorted(set(flat + manual))
                except Exception as e:
                    print(f"  [briefing] {sym}: {e}")
            _briefing_day = _et_now().date()

        scan(client, args.symbols, tracker, key_levels, pm, em, lm)

        # Telegram heartbeat so you know it's alive on your phone
        hb_interval = HEARTBEAT_INTERVAL if _market_open() else HEARTBEAT_CLOSED_INTERVAL
        if time.time() - _last_heartbeat >= hb_interval:
            if not _market_open():
                try:
                    _get_tg().send_raw(
                        f"🤖 <b>QuantDesk alive</b> — market closed  |  {now_str}"
                    )
                except Exception:
                    pass
                _last_heartbeat = time.time()
                time.sleep(args.interval)
                continue
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
                    f"{pm.status if pm else 'watching 👀'}"
                )
                _last_heartbeat = time.time()
            except Exception:
                pass

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
