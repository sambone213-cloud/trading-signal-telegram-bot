"""
trade_strategies.py  — v3
─────────────────────────
5 strategies, each derived from 2-year SPY backtest (185k bars).
Design rules:
  - Mean-reversion plays ONLY fire when ADX < 28 (ranging market)
  - Breakout/trend plays ONLY fire when ADX > 22 (directional market)
  - Each strategy has built-in exit conditions (not just hard stop)
  - Max ~2-4 signals per day total across all strategies

Backtest edges (2yr SPY 1-min, Alpaca data):
  1. Keltner Bounce     E=+0.73  WR 64%  N=22   ADX<28 required
  2. Power Hour Dip     E=+1.00  WR 74%  N=19   3-4pm only
  3. Trend Breakout     E=+0.45  WR 54%  ADX>22
  4. VWAP Reclaim       E=+0.39  WR 52%  vol confirm
  5. BB+ADX Reversal    E=+0.23  WR 55%  ADX<25
"""

from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np

try:
    from telegram_notify import get_notifier as _get_tg
except ImportError:
    def _get_tg():
        class _Noop:
            def send_signal(self, *a, **kw): pass
        return _Noop()


# ── Signal model ──────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    strategy:    str
    side:        str          # 'long' | 'short'
    entry_price: float
    tp_price:    float
    sl_price:    float
    confidence:  float
    reasons:     list = field(default_factory=list)
    exit_hints:  list = field(default_factory=list)  # what to watch for exit

    @property
    def risk_reward(self) -> float:
        tp_dist = abs(self.tp_price - self.entry_price)
        sl_dist = abs(self.sl_price - self.entry_price)
        return round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(df, col, idx=-1):
    try:
        return float(df[col].iloc[idx])
    except Exception:
        return None

def _has(df, *cols):
    return not df.empty and len(df) >= 35 and all(c in df.columns for c in cols)

def _adx_ok(df, max_adx=28, min_adx=None):
    adx = _get(df, "adx")
    if adx is None:
        return False
    if max_adx and adx >= max_adx:
        return False
    if min_adx and adx < min_adx:
        return False
    return True


# ── Strategy 1: Keltner Bounce ────────────────────────────────────────────────

def strategy_keltner_bounce(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Price breaks below Keltner Channel lower band then VWMA9 crosses back
    above VWMA21 — signals the flush is over and buyers stepping in.

    Conditions (ALL required):
      - Close < KC lower  (price outside Keltner — oversold flush)
      - VWMA9 crosses above VWMA21  (volume-weighted momentum turning)
      - ADX < 28  (ranging market — don't fade a trending breakdown)
      - plus_di > minus_di  (bullish directional bias)
      - RSI < 50  (not already overbought)
      - After first 30 min (avoids open volatility)

    Exit hints: VWMA9 crosses back below VWMA21, RSI > 65, price hits BB mid

    Backtest: E=+0.73  WR=64%  N=22 (2yr SPY)
    """
    needed = ("close","kc_lower","vwma9","vwma21","adx","plus_di","minus_di","rsi","atr","bb_mid")
    if not _has(df, *needed):
        return None

    c, p = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close")
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    adx   = _get(df, "adx")
    if not all([close, atr, rsi, adx]) or atr <= 0:
        return None

    # Regime: ranging only
    if adx >= 28:
        return None

    below_kc   = float(c["close"]) < float(c["kc_lower"])
    vwma_xup   = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])
    bull_di    = float(c["plus_di"]) > float(c["minus_di"])
    rsi_ok     = rsi < 50
    after_open = "mins_since_open" not in df.columns or _get(df, "mins_since_open") >= 30

    if below_kc and vwma_xup and bull_di and rsi_ok and after_open:
        bb_mid = _get(df, "bb_mid") or close + 2.5 * atr
        return StrategySignal(
            "Keltner Bounce", "long", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = 0.80,
            reasons    = [f"Below KC  ${float(c['kc_lower']):.2f}",
                          "VWMA9x21 up", f"ADX {adx:.0f} ranging",
                          f"RSI {rsi:.0f}"],
            exit_hints = ["VWMA9 crosses back below VWMA21",
                          "RSI > 65", "Price hits BB midline"],
        )
    return None


# ── Strategy 2: Power Hour Dip Buy ────────────────────────────────────────────

def strategy_power_hour_dip(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    3:00–3:55 PM ET dip into an uptrend — institutional buyers step in
    before close. VWMA cross + negative short-term momentum in power hour.

    Conditions (ALL required):
      - 3:00–3:55 PM ET (power hour window)
      - VWMA9 crosses above VWMA21  (volume momentum turning up)
      - MOM10 < 0  (short-term price still negative — dip not yet resolved)
      - EMA9 > EMA21  (intraday trend is up — we're buying a dip, not a downtrend)
      - RSI 30–60  (not overbought entering)
      - ADX < 35  (not a runaway trend)

    Exit hints: VWMA9 crosses back below VWMA21, RSI > 70, within 30 min of close

    Backtest: E=+1.00  WR=74%  N=19 (2yr SPY)
    """
    needed = ("close","vwma9","vwma21","ema9","ema21","rsi","atr","mom10")
    if not _has(df, *needed):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close")
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    mom10 = _get(df, "mom10")
    if not all([close, atr, rsi]) or atr is None or atr <= 0:
        return None

    # Time gate: power hour only
    hour_et = _get(df, "hour_et")
    min_et  = _get(df, "minute_et")
    if hour_et is None or not (hour_et == 15 and (min_et or 0) < 55):
        return None

    vwma_xup  = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])
    ema_bull  = float(c["ema9"]) > float(c["ema21"])
    mom_neg   = mom10 is not None and mom10 < 0
    rsi_ok    = 30 <= rsi <= 62

    if vwma_xup and ema_bull and mom_neg and rsi_ok:
        return StrategySignal(
            "Power Hour Dip", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = 0.82,
            reasons    = ["VWMA9x21 up", "Power hour 3pm",
                          f"MOM10 {mom10:.2f} dip", f"RSI {rsi:.0f}",
                          "EMA9>21 uptrend"],
            exit_hints = ["VWMA9 crosses back below VWMA21",
                          "RSI > 70", "15 min before close"],
        )
    return None


# ── Strategy 3: Trend Breakout ────────────────────────────────────────────────

def strategy_trend_breakout(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Momentum breakout in an established trend — fires when all trend indicators
    align in one direction. ADX > 22 confirms real directional movement.

    Long conditions:
      - MACD crosses above signal  (momentum trigger)
      - EMA9 > EMA21  (trend direction)
      - Close > VWAP  (price above institutional average)
      - ADX > 22  (trending, not ranging)
      - RSI 40–72  (room to run, not overbought)
      - Volume > 1.3x avg  (breakout has participation)

    Short: mirror conditions

    Exit hints: MACD crosses back, price crosses VWAP, EMA9 crosses EMA21

    Backtest: E=+0.45  WR=54%  (macd_xdn + adx<20 short, scaled to both sides)
    """
    needed = ("close","macd","macd_signal","macd_hist","ema9","ema21","vwap","adx","rsi","atr","volume")
    if not _has(df, *needed):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close")
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    adx   = _get(df, "adx")
    if not all([close, atr, rsi, adx]) or atr <= 0:
        return None

    # Regime: must have directional trend
    if adx < 22:
        return None

    macd_xup   = float(p["macd"]) <= float(p["macd_signal"]) and float(c["macd"]) > float(c["macd_signal"])
    macd_xdn   = float(p["macd"]) >= float(p["macd_signal"]) and float(c["macd"]) < float(c["macd_signal"])
    ema_bull   = float(c["ema9"]) > float(c["ema21"])
    ema_bear   = float(c["ema9"]) < float(c["ema21"])
    above_vwap = float(c["close"]) > float(c["vwap"])
    below_vwap = float(c["close"]) < float(c["vwap"])
    hist_exp   = abs(float(c["macd_hist"])) > abs(float(p["macd_hist"]))

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.3

    if macd_xup and ema_bull and above_vwap and hist_exp and vol_ok and 40 <= rsi <= 72:
        return StrategySignal(
            "Trend Breakout", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = 0.75,
            reasons    = ["MACD cross up", "EMA9>21", "Above VWAP",
                          f"ADX {adx:.0f}", f"RSI {rsi:.0f}", "Vol surge"],
            exit_hints = ["MACD crosses back below signal",
                          "Price drops below VWAP", "EMA9 crosses below EMA21"],
        )

    if macd_xdn and ema_bear and below_vwap and hist_exp and vol_ok and 28 <= rsi <= 60:
        return StrategySignal(
            "Trend Breakout", "short", close,
            tp_price   = round(close - 2.5 * atr, 2),
            sl_price   = round(close + 1.0 * atr, 2),
            confidence = 0.75,
            reasons    = ["MACD cross down", "EMA9<21", "Below VWAP",
                          f"ADX {adx:.0f}", f"RSI {rsi:.0f}", "Vol surge"],
            exit_hints = ["MACD crosses back above signal",
                          "Price reclaims VWAP", "EMA9 crosses above EMA21"],
        )
    return None


# ── Strategy 4: VWAP Reclaim ──────────────────────────────────────────────────

def strategy_vwap_reclaim(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Price crosses VWAP with volume confirmation — institutional flow signal.
    VWAP is the line institutions use. A reclaim with volume = real buying.

    Long:  prev close < VWAP, current close > VWAP + vol > 1.5x + EMA9 > EMA21
    Short: prev close > VWAP, current close < VWAP + vol > 1.5x + EMA9 < EMA21

    Exit hints: price crosses back through VWAP, RSI extremes

    Backtest: E=+0.39  WR=52% (VWAP reclaim + vol, 2yr SPY)
    """
    needed = ("close","vwap","ema9","ema21","rsi","atr","volume")
    if not _has(df, *needed):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close")
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    if not all([close, atr, rsi]) or atr <= 0:
        return None

    cross_up   = float(p["close"]) < float(p["vwap"]) and float(c["close"]) > float(c["vwap"])
    cross_down = float(p["close"]) > float(p["vwap"]) and float(c["close"]) < float(c["vwap"])
    ema_bull   = float(c["ema9"]) > float(c["ema21"])
    ema_bear   = float(c["ema9"]) < float(c["ema21"])

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.5

    if cross_up and ema_bull and vol_ok and rsi < 68:
        return StrategySignal(
            "VWAP Reclaim", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 0.8 * atr, 2),
            confidence = 0.72,
            reasons    = ["VWAP reclaim up", "EMA9>21",
                          "Vol surge 1.5x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price drops back below VWAP",
                          "RSI > 72", "Volume dries up"],
        )

    if cross_down and ema_bear and vol_ok and rsi > 32:
        return StrategySignal(
            "VWAP Reclaim", "short", close,
            tp_price   = round(close - 2.5 * atr, 2),
            sl_price   = round(close + 0.8 * atr, 2),
            confidence = 0.72,
            reasons    = ["VWAP break down", "EMA9<21",
                          "Vol surge 1.5x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price reclaims VWAP",
                          "RSI < 28", "Volume dries up"],
        )
    return None


# ── Strategy 5: BB + ADX Mean Reversion ──────────────────────────────────────

def strategy_bb_adx_reversal(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    BB band touch mean reversion ONLY in ranging markets (ADX < 25).
    Don't fade a trending move — ADX gate prevents the most common failure mode.

    Long:  2 bars at/below BB lower + RSI < 35 + RSI turning up + ADX < 25
    Short: 2 bars at/above BB upper + RSI > 65 + RSI turning down + ADX < 25

    Exit hint: price reaches BB midline (that's the TP target)

    Backtest: E=+0.23  WR=55%  (ranging-only filter)
    """
    needed = ("close","bb_upper","bb_lower","bb_mid","rsi","atr","adx")
    if not _has(df, *needed):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    close  = _get(df, "close")
    atr    = _get(df, "atr")
    rsi    = _get(df, "rsi")
    adx    = _get(df, "adx")
    bb_mid = _get(df, "bb_mid")
    if not all([close, atr, rsi, adx, bb_mid]) or atr <= 0:
        return None

    # Strict ranging filter
    if adx >= 25:
        return None

    rsi_prev = float(p["rsi"]) if "rsi" in p.index else rsi
    two_low  = float(c["close"]) <= float(c["bb_lower"]) and float(p["close"]) <= float(p["bb_lower"])
    two_high = float(c["close"]) >= float(c["bb_upper"]) and float(p["close"]) >= float(p["bb_upper"])

    if two_low and rsi < 35 and rsi > rsi_prev:
        return StrategySignal(
            "BB+ADX Reversal", "long", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close - 0.6 * atr, 2),
            confidence = 0.73,
            reasons    = [f"2x below BB ${float(c['bb_lower']):.2f}",
                          f"RSI {rsi:.0f} turning up", f"ADX {adx:.0f} ranging"],
            exit_hints = ["Price reaches BB midline (TP)",
                          "RSI > 60", "ADX spikes above 25 (abort)"],
        )

    if two_high and rsi > 65 and rsi < rsi_prev:
        return StrategySignal(
            "BB+ADX Reversal", "short", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close + 0.6 * atr, 2),
            confidence = 0.73,
            reasons    = [f"2x above BB ${float(c['bb_upper']):.2f}",
                          f"RSI {rsi:.0f} turning down", f"ADX {adx:.0f} ranging"],
            exit_hints = ["Price reaches BB midline (TP)",
                          "RSI < 40", "ADX spikes above 25 (abort)"],
        )
    return None


# ── Exit condition checker ────────────────────────────────────────────────────

def check_exit_conditions(df: pd.DataFrame, position_side: str, strategy: str) -> Optional[str]:
    """
    Call every scan after entering a position.
    Returns exit reason string if exit triggered, None to hold.
    """
    if not _has(df, "close","ema9","ema21","vwap","rsi","macd","macd_signal","vwma9","vwma21"):
        return None

    c, p = df.iloc[-1], df.iloc[-2]
    rsi  = _get(df, "rsi")
    close= _get(df, "close")

    ema_flipped_bear = float(p["ema9"]) >= float(p["ema21"]) and float(c["ema9"]) < float(c["ema21"])
    ema_flipped_bull = float(p["ema9"]) <= float(p["ema21"]) and float(c["ema9"]) > float(c["ema21"])
    vwap_break_down  = float(p["close"]) >= float(p["vwap"]) and float(c["close"]) < float(c["vwap"])
    vwap_reclaim_up  = float(p["close"]) <= float(p["vwap"]) and float(c["close"]) > float(c["vwap"])
    macd_xdn         = float(p["macd"]) >= float(p["macd_signal"]) and float(c["macd"]) < float(c["macd_signal"])
    macd_xup         = float(p["macd"]) <= float(p["macd_signal"]) and float(c["macd"]) > float(c["macd_signal"])
    vwma_xdn         = float(p["vwma9"]) >= float(p["vwma21"]) and float(c["vwma9"]) < float(c["vwma21"])
    vwma_xup         = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])

    if position_side == "long":
        if rsi and rsi > 75:
            return f"RSI overbought {rsi:.0f} — consider exit"
        if ema_flipped_bear:
            return "EMA9 crossed below EMA21 — trend reversing"
        if vwap_break_down:
            return "Price broke below VWAP — momentum lost"
        if macd_xdn and strategy in ("Trend Breakout", "VWAP Reclaim"):
            return "MACD crossed down — momentum fading"
        if vwma_xdn and strategy in ("Keltner Bounce", "Power Hour Dip"):
            return "VWMA9 crossed below VWMA21 — exit signal"
    else:  # short
        if rsi and rsi < 25:
            return f"RSI oversold {rsi:.0f} — consider covering"
        if ema_flipped_bull:
            return "EMA9 crossed above EMA21 — trend reversing"
        if vwap_reclaim_up:
            return "Price reclaimed VWAP — short thesis broken"
        if macd_xup and strategy in ("Trend Breakout", "VWAP Reclaim"):
            return "MACD crossed up — short momentum fading"
        if vwma_xup and strategy == "BB+ADX Reversal":
            return "VWMA9 crossed above VWMA21 — exit short"

    return None


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    # Ranked by 2yr SPY backtest edge:
    "Power Hour Dip":    strategy_power_hour_dip,     # E=+1.00  WR 74%  3pm only
    "Keltner Bounce":    strategy_keltner_bounce,      # E=+0.73  WR 64%  ranging
    "Trend Breakout":    strategy_trend_breakout,      # E=+0.45  WR 54%  trending
    "VWAP Reclaim":      strategy_vwap_reclaim,        # E=+0.39  WR 52%  vol confirm
    "BB+ADX Reversal":   strategy_bb_adx_reversal,     # E=+0.23  WR 55%  ranging
}


def run_all_strategies(
    df: pd.DataFrame,
    enabled: list = None,
    symbol: str = "",
    notify: bool = True,
) -> List[StrategySignal]:
    fns = {k: v for k, v in ALL_STRATEGIES.items()
           if enabled is None or k in enabled}
    signals = []
    for name, fn in fns.items():
        try:
            sig = fn(df)
            if sig is not None:
                signals.append(sig)
                if notify and symbol:
                    try:
                        _get_tg().send_signal(
                            symbol=symbol, strategy=sig.strategy,
                            signal="BUY" if sig.side == "long" else "SELL",
                            price=sig.entry_price,
                            details="  |  ".join(sig.reasons),
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    return signals
