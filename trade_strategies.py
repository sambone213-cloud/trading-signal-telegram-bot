"""
trade_strategies.py  — v5
─────────────────────────
10 strategies. Original 7 from the 2-year SPY backtest; 3 new (Jun 2026)
walk-forward validated on 17 months of SPY 1-min via backtest_explorer.py
(128k bars, Alpaca IEX, 3 rolling train/test windows, 1,426 candidates).

Changes from v4:
  - NEW Oversold Dip Buy:  RSI<25 panic + EMA21>EMA50 — held-out edge beat training
  - NEW Opening Drive:     9:45-10:00 DI-direction trade — most stable setup in sweep
  - NEW Lunch VWAP Hold:   12pm above-VWAP long — short variant tested weak, excluded

Strategy rank by backtest edge:
  1. Power Hour Dip      E=+1.00  WR 74%   3-4pm dip buy in uptrend
  2. Keltner Bounce      E=+0.73  WR 64%   KC break reversal, ADX<28
  3. Momentum Flip       E=+0.55  WR 62%   EMA+MACD flip with volume — fires daily
  4. Trend Breakout      E=+0.45  WR 54%   MACD+EMA+VWAP, ADX>22
  5. VWAP Reclaim        E=+0.39  WR 52%   VWAP cross + volume
  6. Oversold Dip Buy    E=+0.37  WR 49%   RSI<25 flush in uptrend (walk-forward)
  7. Opening Drive       E=+0.31  WR 50%   first-30min DI direction (walk-forward)
  8. Lunch VWAP Hold     E=+0.31  WR 49%   12pm VWAP hold, long only (walk-forward)
  9. BB+ADX Reversal     E=+0.23  WR 55%   BB touch, ranging only
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
    side:        str
    entry_price: float
    tp_price:    float
    sl_price:    float
    confidence:  float
    reasons:     list = field(default_factory=list)
    exit_hints:  list = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        tp = abs(self.tp_price - self.entry_price)
        sl = abs(self.sl_price - self.entry_price)
        return round(tp / sl, 2) if sl > 0 else 0.0

    @property
    def conf_label(self) -> str:
        c = self.confidence
        filled = int(round(c * 5))
        bar    = ("*" * filled).ljust(5, "-")
        tier   = "HIGH" if c >= 0.80 else ("MED " if c >= 0.72 else "LOW ")
        return f"{tier} [{bar}] {c*100:.0f}%"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(df, col, idx=-1):
    try:   return float(df[col].iloc[idx])
    except: return None

def _has(df, *cols):
    return not df.empty and len(df) >= 35 and all(c in df.columns for c in cols)

def _prev(df, col):
    try:   return float(df[col].iloc[-2])
    except: return None


# ── Market regime engine ──────────────────────────────────────────────────────
# Both of Sam's losing days (Jun 13, Jun 15) were chop. Trend-following setups
# fired into directionless tape and bled to their stops. The regime layer scores
# each signal's fit to current conditions so chop signals are visibly low-conviction.

TREND_STRATEGIES = {"Momentum Flip", "Trend Breakout", "ORB Breakout",
                    "Opening Drive", "VWAP Reclaim", "Power Hour Dip"}
MEANREV_STRATEGIES = {"Keltner Bounce", "BB+ADX Reversal",
                      "Oversold Dip Buy", "Lunch VWAP Hold"}


def price_efficiency(df, n: int = 20) -> float:
    """
    Net move / total path over last n bars. ~1.0 = clean directional move,
    near 0 = chop (lots of motion, no progress). Regime measure that does NOT
    depend on ADX — ADX is unreliable for ~15 bars after an overnight gap
    (Jun 15 read ADX 87 at 9:45 because the 740->753 gap spiked the DI calc).
    """
    try:
        c = df["close"].tail(n + 1)
        if len(c) < n + 1:
            return 0.5
        net  = abs(float(c.iloc[-1]) - float(c.iloc[0]))
        path = float(c.diff().abs().sum())
        return net / path if path > 0 else 0.0
    except Exception:
        return 0.5


def market_regime(df) -> str:
    """TREND_UP | TREND_DOWN | CHOP | MIXED from price efficiency + EMA stack."""
    eff   = price_efficiency(df)
    ema9  = _get(df, "ema9"); ema21 = _get(df, "ema21"); ema50 = _get(df, "ema50")
    mins  = _get(df, "mins_since_open")
    if None in (ema9, ema21, ema50):
        return "MIXED"
    # First 15 min after the open: ADX/DI unstable on gap days, don't label TREND
    if mins is not None and mins < 15:
        return "MIXED"
    if eff >= 0.38 and ema9 > ema21 > ema50:
        return "TREND_UP"
    if eff >= 0.38 and ema9 < ema21 < ema50:
        return "TREND_DOWN"
    if eff < 0.22:
        return "CHOP"
    return "MIXED"


def regime_fit(strategy: str, regime: str) -> str:
    """'good' | 'poor' | 'neutral' — does the strategy's type suit the regime?"""
    if regime in ("TREND_UP", "TREND_DOWN"):
        return "good" if strategy in TREND_STRATEGIES else "poor"
    if regime == "CHOP":
        return "good" if strategy in MEANREV_STRATEGIES else "poor"
    return "neutral"


# ── Dynamic confidence scorer ─────────────────────────────────────────────────

def _score_confidence(df, base: float, side: str) -> float:
    """
    Adjusts base confidence up/down based on current market conditions.
    Returns value clamped to [0.55, 0.95].
    """
    score = base
    adx      = _get(df, "adx")      or 20
    vol_r    = _get(df, "vol_ratio") or 1.0
    rsi      = _get(df, "rsi")       or 50
    hour_et  = _get(df, "hour_et")   or 12

    # ADX: strong trend = more conviction on breakouts, less on reversals
    if adx > 30:  score += 0.05
    elif adx < 15: score -= 0.05

    # Volume: high participation = more reliable
    if vol_r > 2.0:   score += 0.06
    elif vol_r > 1.5: score += 0.03
    elif vol_r < 0.8: score -= 0.04

    # RSI distance from extreme (room to run)
    if side == "long":
        room = 80 - rsi
        if room > 25:  score += 0.04
        elif room < 10: score -= 0.06   # overbought entry
    else:
        room = rsi - 20
        if room > 25:  score += 0.04
        elif room < 10: score -= 0.06   # oversold short entry

    # Best trading hours: 10:30-14:00 ET
    if 10 <= hour_et <= 13:  score += 0.03
    elif hour_et >= 15:       score += 0.02   # power hour still good
    elif hour_et == 9:        score -= 0.05   # first 30 min — noise

    return round(max(0.55, min(0.95, score)), 2)


# ── Strategy 1: Keltner Bounce ────────────────────────────────────────────────

def strategy_keltner_bounce(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Price breaks below Keltner Channel lower band then VWMA9 crosses back
    above VWMA21. Ranging markets only (ADX < 28).
    Backtest: E=+0.73  WR=64%
    """
    if not _has(df, "close","kc_lower","vwma9","vwma21","adx","plus_di","minus_di","rsi","atr","bb_mid"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi"); adx = _get(df, "adx")
    if not all([close, atr, rsi, adx]) or atr <= 0: return None
    if adx >= 28: return None

    below_kc  = float(c["close"]) < float(c["kc_lower"])
    vwma_xup  = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])
    bull_di   = float(c["plus_di"]) > float(c["minus_di"])
    after_open = _get(df, "mins_since_open") is None or (_get(df, "mins_since_open") or 0) >= 30

    if below_kc and vwma_xup and bull_di and rsi < 50 and after_open:
        bb_mid = _get(df, "bb_mid") or close + 2.5 * atr
        conf   = _score_confidence(df, 0.78, "long")
        return StrategySignal(
            "Keltner Bounce", "long", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"Below KC ${float(c['kc_lower']):.2f}", "VWMA9x21 up",
                          f"ADX {adx:.0f} ranging", f"RSI {rsi:.0f}"],
            exit_hints = ["VWMA9 crosses back below VWMA21",
                          "RSI > 65", "Price hits BB midline (TP)"],
        )
    return None


# ── Strategy 2: Power Hour Dip ────────────────────────────────────────────────

def strategy_power_hour_dip(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    3:00-3:55 PM ET dip buy — VWMA cross up + negative short-term momentum
    in an established uptrend. Institutional close-of-day buying.
    Backtest: E=+1.00  WR=74%
    """
    if not _has(df, "close","vwma9","vwma21","ema9","ema21","rsi","atr","mom10"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi"); mom10 = _get(df, "mom10")
    hour_et = _get(df, "hour_et"); min_et = _get(df, "minute_et")
    if not all([close, atr, rsi]) or atr <= 0: return None
    if hour_et is None or not (hour_et == 15 and (min_et or 0) < 55): return None

    vwma_xup = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])
    ema_bull  = float(c["ema9"]) > float(c["ema21"])
    mom_neg   = mom10 is not None and mom10 < 0

    if vwma_xup and ema_bull and mom_neg and 30 <= rsi <= 62:
        conf = _score_confidence(df, 0.80, "long")
        return StrategySignal(
            "Power Hour Dip", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = ["VWMA9x21 up", "Power hour 3pm",
                          f"MOM10 dip {mom10:.2f}", f"RSI {rsi:.0f}", "EMA9>21"],
            exit_hints = ["VWMA9 crosses back below VWMA21",
                          "RSI > 70", "15 min before close"],
        )
    return None


# ── Strategy 3: Momentum Flip ─────────────────────────────────────────────────

def strategy_momentum_flip(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    EMA9/21 cross AND MACD cross in the same direction WITH volume surge.
    One cross triggers, the other must confirm within a 3-bar window — EMA and
    MACD almost never align on the exact same 1-min bar.

    Long:  EMA9 recently crossed above EMA21 + MACD currently bullish (or vice versa)
           + vol > 1.5x + RSI 35-65 + both currently aligned
    Short: mirror image

    Exit: opposite EMA cross OR MACD crosses back AND vol confirms
    Backtest: E=+0.55  WR=62%
    """
    if not _has(df, "close","ema9","ema21","macd","macd_signal","macd_hist","rsi","atr","vwap","volume"):
        return None
    c     = df.iloc[-1]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    if not all([close, atr, rsi]) or atr <= 0: return None

    mins = _get(df, "mins_since_open") or 999
    if mins < 30: return None

    # Check for a cross anywhere in the last 3 bars
    lb = min(3, len(df) - 1)

    def _recent_xup(col_a, col_b):
        for i in range(len(df) - lb, len(df)):
            if i > 0 and df[col_a].iloc[i-1] <= df[col_b].iloc[i-1] and df[col_a].iloc[i] > df[col_b].iloc[i]:
                return True
        return False

    def _recent_xdn(col_a, col_b):
        for i in range(len(df) - lb, len(df)):
            if i > 0 and df[col_a].iloc[i-1] >= df[col_b].iloc[i-1] and df[col_a].iloc[i] < df[col_b].iloc[i]:
                return True
        return False

    ema_xup_recent  = _recent_xup("ema9", "ema21")
    ema_xdn_recent  = _recent_xdn("ema9", "ema21")
    macd_xup_recent = _recent_xup("macd", "macd_signal")
    macd_xdn_recent = _recent_xdn("macd", "macd_signal")

    # Both must currently agree on direction
    ema_bull_now  = float(c["ema9"]) > float(c["ema21"])
    ema_bear_now  = float(c["ema9"]) < float(c["ema21"])
    macd_bull_now = float(c["macd"]) > float(c["macd_signal"])
    macd_bear_now = float(c["macd"]) < float(c["macd_signal"])

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(c["volume"]) / vol_ma if vol_ma > 0 else 1.0
    vol_ok = vol_r >= 1.5   # lowered from 2.0 — 2x was rarely hit on SPY 1-min bars

    if (ema_xup_recent or macd_xup_recent) and ema_bull_now and macd_bull_now and vol_ok and 35 <= rsi <= 65:
        trigger = "EMA9 crosses above EMA21" if ema_xup_recent else "MACD cross up"
        confirm = "MACD bull" if ema_xup_recent else "EMA9>21"
        conf = _score_confidence(df, 0.76, "long")
        return StrategySignal(
            "Momentum Flip", "long", close,
            tp_price   = round(close + 3.0 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [trigger, confirm,
                          f"Vol {vol_r:.1f}x surge", f"RSI {rsi:.0f}"],
            exit_hints = ["EMA9 crosses back below EMA21",
                          "MACD crosses down with vol > 1.5x",
                          "RSI > 75"],
        )

    if (ema_xdn_recent or macd_xdn_recent) and ema_bear_now and macd_bear_now and vol_ok and 35 <= rsi <= 65:
        trigger = "EMA9 crosses below EMA21" if ema_xdn_recent else "MACD cross down"
        confirm = "MACD bear" if ema_xdn_recent else "EMA9<21"
        conf = _score_confidence(df, 0.76, "short")
        return StrategySignal(
            "Momentum Flip", "short", close,
            tp_price   = round(close - 3.0 * atr, 2),
            sl_price   = round(close + 1.0 * atr, 2),
            confidence = conf,
            reasons    = [trigger, confirm,
                          f"Vol {vol_r:.1f}x surge", f"RSI {rsi:.0f}"],
            exit_hints = ["EMA9 crosses back above EMA21",
                          "MACD crosses up with vol > 1.5x",
                          "RSI < 25"],
        )
    return None


# ── Strategy 4: Trend Breakout ────────────────────────────────────────────────

def strategy_trend_breakout(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    MACD cross + EMA alignment + above/below VWAP + volume in trending market.
    Exit requires MACD reversal + volume confirmation (prevents 2-min exits).
    ADX > 22 required — no trading in flat markets.
    Backtest: E=+0.45  WR=54%
    """
    if not _has(df, "close","macd","macd_signal","macd_hist","ema9","ema21","vwap","adx","rsi","atr","volume"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi"); adx = _get(df, "adx")
    if not all([close, atr, rsi, adx]) or atr <= 0: return None
    if adx < 22: return None

    macd_xup   = float(p["macd"]) <= float(p["macd_signal"]) and float(c["macd"]) > float(c["macd_signal"])
    macd_xdn   = float(p["macd"]) >= float(p["macd_signal"]) and float(c["macd"]) < float(c["macd_signal"])
    ema_bull   = float(c["ema9"]) > float(c["ema21"])
    ema_bear   = float(c["ema9"]) < float(c["ema21"])
    above_vwap = float(c["close"]) > float(c["vwap"])
    below_vwap = float(c["close"]) < float(c["vwap"])
    hist_exp   = abs(float(c["macd_hist"])) > abs(float(p["macd_hist"]))

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(c["volume"]) / vol_ma if vol_ma > 0 else 1.0
    vol_ok = vol_r >= 1.4

    # Don't fire if EMA cross just happened (Momentum Flip covers that)
    ema_just_crossed = (abs(float(c["ema9"]) - float(c["ema21"])) < atr * 0.1)
    if ema_just_crossed: return None

    if macd_xup and ema_bull and above_vwap and hist_exp and vol_ok and 40 <= rsi <= 72:
        conf = _score_confidence(df, 0.74, "long")
        return StrategySignal(
            "Trend Breakout", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = ["MACD cross up", "EMA9>21", "Above VWAP",
                          f"ADX {adx:.0f}", f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["MACD crosses down AND vol > 1.5x (both required)",
                          "Price closes below VWAP", "EMA9 crosses below EMA21"],
        )

    if macd_xdn and ema_bear and below_vwap and hist_exp and vol_ok and 28 <= rsi <= 60:
        conf = _score_confidence(df, 0.74, "short")
        return StrategySignal(
            "Trend Breakout", "short", close,
            tp_price   = round(close - 2.5 * atr, 2),
            sl_price   = round(close + 1.0 * atr, 2),
            confidence = conf,
            reasons    = ["MACD cross down", "EMA9<21", "Below VWAP",
                          f"ADX {adx:.0f}", f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["MACD crosses up AND vol > 1.5x (both required)",
                          "Price reclaims VWAP", "EMA9 crosses above EMA21"],
        )
    return None


# ── Strategy 5: VWAP Reclaim ──────────────────────────────────────────────────

def strategy_vwap_reclaim(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Price crosses VWAP with volume and EMA alignment.
    Volume threshold lowered to 1.3x (from 1.5x) so it fires more consistently.
    Backtest: E=+0.39  WR=52%
    """
    if not _has(df, "close","vwap","ema9","ema21","rsi","atr","volume"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    if not all([close, atr, rsi]) or atr <= 0: return None

    cross_up   = float(p["close"]) < float(p["vwap"]) and float(c["close"]) > float(c["vwap"])
    cross_down = float(p["close"]) > float(p["vwap"]) and float(c["close"]) < float(c["vwap"])
    ema_bull   = float(c["ema9"]) > float(c["ema21"])
    ema_bear   = float(c["ema9"]) < float(c["ema21"])

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(c["volume"]) / vol_ma if vol_ma > 0 else 1.0
    vol_ok = vol_r >= 1.3

    if cross_up and ema_bull and vol_ok and rsi < 68:
        conf = _score_confidence(df, 0.70, "long")
        return StrategySignal(
            "VWAP Reclaim", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 0.8 * atr, 2),
            confidence = conf,
            reasons    = ["VWAP reclaim up", "EMA9>21",
                          f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price drops back below VWAP",
                          "RSI > 72", "EMA9 crosses below EMA21"],
        )

    if cross_down and ema_bear and vol_ok and rsi > 32:
        conf = _score_confidence(df, 0.70, "short")
        return StrategySignal(
            "VWAP Reclaim", "short", close,
            tp_price   = round(close - 2.5 * atr, 2),
            sl_price   = round(close + 0.8 * atr, 2),
            confidence = conf,
            reasons    = ["VWAP break down", "EMA9<21",
                          f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price reclaims VWAP",
                          "RSI < 28", "EMA9 crosses above EMA21"],
        )
    return None


# ── Strategy 6: BB + ADX Mean Reversion ──────────────────────────────────────

def strategy_bb_adx_reversal(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    BB band touch mean reversion in ranging markets only (ADX < 25).
    Backtest: E=+0.23  WR=55%
    """
    if not _has(df, "close","bb_upper","bb_lower","bb_mid","rsi","atr","adx"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    adx   = _get(df, "adx"); bb_mid = _get(df, "bb_mid")
    ema9  = _get(df, "ema9");  ema21 = _get(df, "ema21")
    if not all([close, atr, rsi, adx, bb_mid, ema9, ema21]) or atr <= 0: return None
    if adx >= 30: return None   # relaxed from 25 — catches oversold/overbought extremes in mild trends

    rsi_prev = float(p["rsi"]) if "rsi" in p.index else rsi
    two_low  = float(c["close"]) <= float(c["bb_lower"]) and float(p["close"]) <= float(p["bb_lower"])
    two_high = float(c["close"]) >= float(c["bb_upper"]) and float(p["close"]) >= float(p["bb_upper"])

    ema_bull = ema9 > ema21
    ema_bear = ema9 < ema21

    # Don't fire counter-trend: no SHORT in uptrend, no LONG in downtrend
    if two_low and ema_bear: return None   # price below BB but EMA already bearish — not a reversal setup
    if two_high and ema_bull: return None  # price above BB in a bull trend — don't fade it

    if two_low and rsi < 35 and rsi > rsi_prev:
        conf = _score_confidence(df, 0.70, "long")
        return StrategySignal(
            "BB+ADX Reversal", "long", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close - 0.6 * atr, 2),
            confidence = conf,
            reasons    = [f"2x below BB ${float(c['bb_lower']):.2f}",
                          f"RSI {rsi:.0f} turning up", f"ADX {adx:.0f} ranging"],
            exit_hints = ["Price reaches BB midline (TP target)",
                          "RSI > 60", "ADX spikes above 25 — abort"],
        )

    if two_high and rsi > 65 and rsi < rsi_prev:
        conf = _score_confidence(df, 0.70, "short")
        return StrategySignal(
            "BB+ADX Reversal", "short", close,
            tp_price   = round(bb_mid, 2),
            sl_price   = round(close + 0.6 * atr, 2),
            confidence = conf,
            reasons    = [f"2x above BB ${float(c['bb_upper']):.2f}",
                          f"RSI {rsi:.0f} turning down", f"ADX {adx:.0f} ranging"],
            exit_hints = ["Price reaches BB midline (TP target)",
                          "RSI < 40", "ADX spikes above 25 — abort"],
        )
    return None


# ── Exit condition checker ────────────────────────────────────────────────────

def check_exit_conditions(df: pd.DataFrame, position_side: str, strategy: str,
                           bars_held: int = 0) -> Optional[str]:
    """
    Returns exit reason string if exit triggered, None to hold.
    bars_held: how many bars since entry (prevents premature exits on Trend Breakout).
    """
    if not _has(df, "close","ema9","ema21","vwap","rsi","macd","macd_signal","vwma9","vwma21","volume"):
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

    # Volume confirmation for MACD exits (prevents 2-min false exits)
    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(c["volume"]) / vol_ma if vol_ma > 0 else 1.0
    macd_vol_confirmed = vol_r >= 1.5

    # Minimum hold: Trend Breakout needs 10 bars before MACD exit is valid
    macd_min_hold = bars_held >= 10 or strategy not in ("Trend Breakout",)

    # Momentum Flip: EMA cross triggered entry, so EMA cross alone isn't a valid exit.
    # Exit only on MACD reversal (with vol) or VWAP flip or RSI extreme.
    is_flip = strategy == "Momentum Flip"

    if position_side == "long":
        if rsi and rsi > 60 and strategy == "Oversold Dip Buy":
            return f"RSI recovered to {rsi:.0f} — dip-buy target reached"
        if rsi and rsi > 78:
            return f"RSI overbought {rsi:.0f} — exit"
        if ema_flipped_bear and not is_flip:
            return "EMA9 crossed below EMA21 — trend reversed"
        if vwap_break_down and strategy in ("Trend Breakout", "VWAP Reclaim", "Momentum Flip",
                                            "Opening Drive", "Lunch VWAP Hold"):
            return "Price broke below VWAP — momentum lost"
        if macd_xdn and macd_vol_confirmed and macd_min_hold:
            return f"MACD crossed down with {vol_r:.1f}x vol — exit"
        if vwma_xdn and strategy in ("Keltner Bounce", "Power Hour Dip"):
            return "VWMA9 crossed below VWMA21 — exit"
    else:
        if rsi and rsi < 22:
            return f"RSI oversold {rsi:.0f} — cover"
        if ema_flipped_bull and not is_flip:
            return "EMA9 crossed above EMA21 — trend reversed"
        if vwap_reclaim_up and strategy in ("Trend Breakout", "VWAP Reclaim", "Momentum Flip",
                                            "Opening Drive"):
            return "Price reclaimed VWAP — short thesis broken"
        if macd_xup and macd_vol_confirmed and macd_min_hold:
            return f"MACD crossed up with {vol_r:.1f}x vol — cover"
        if vwma_xup and strategy == "BB+ADX Reversal":
            return "VWMA9 crossed above VWMA21 — cover short"

    return None


# ── Strategy 7: Opening Range Breakout ───────────────────────────────────────

def strategy_orb(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Opening Range Breakout — price breaks above/below the 9:30-10:00 AM
    high/low with volume and EMA confirmation.

    The ORB high/low are the clearest levels on the chart every day.
    When price breaks them with volume, it tends to run 2-3× the opening range.

    Long:  close > ORB high + buffer, EMA9>21, vol>1.5x, not overbought
    Short: close < ORB low  - buffer, EMA9<21, vol>1.5x, not oversold
    SL: 1×ATR from entry (ORB level becomes support/resistance after break)
    TP: entry ± 3×ATR
    Valid: 10:00 AM – 2:30 PM ET only
    """
    if not _has(df, "close","high","low","ema9","ema21","atr","volume","hour_et","minute_et"):
        return None

    # Identify opening range bars: 9:30–9:59 ET — TODAY only. The live feed
    # now carries 2 days of bars, so without a date filter the opening range
    # would span yesterday's open too.
    orb_mask = (df["hour_et"] == 9) & (df["minute_et"] >= 30)
    if "datetime" in df.columns:
        # market-hours bars: UTC date == ET date, so this is safe
        last_date = df["datetime"].iloc[-1].date()
        orb_mask = orb_mask & (df["datetime"].dt.date == last_date)
    orb_bars  = df[orb_mask]
    if len(orb_bars) < 20:   # need at least 20 bars to define the range
        return None

    orb_high = float(orb_bars["high"].max())
    orb_low  = float(orb_bars["low"].min())
    orb_size = orb_high - orb_low
    if orb_size <= 0: return None

    c, p    = df.iloc[-1], df.iloc[-2]
    close   = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    adx     = _get(df, "adx") or 15
    hour_et = _get(df, "hour_et") or 0
    min_et  = _get(df, "minute_et") or 0
    if not all([close, atr, rsi]) or atr <= 0: return None

    # Time gate: ORB must be complete (after 10:00 AM) and not too late (before 2:30 PM)
    mins = _get(df, "mins_since_open") or 0
    if mins < 30: return None
    if hour_et > 14 or (hour_et == 14 and min_et > 30): return None

    ema_bull = float(c["ema9"]) > float(c["ema21"])
    ema_bear = float(c["ema9"]) < float(c["ema21"])

    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(c["volume"]) / vol_ma if vol_ma > 0 else 1.0
    vol_ok = vol_r >= 1.5

    # Look back up to 10 bars for the break — catches moves the bot may have missed
    # when it started late or the scan interval skipped the exact break bar.
    buffer = 0.1 * atr
    lb     = min(10, len(df) - 1)
    break_up = (
        float(c["close"]) > orb_high + buffer and   # still above ORB now
        any(
            df["close"].iloc[i-1] <= orb_high + buffer and df["close"].iloc[i] > orb_high + buffer
            for i in range(len(df) - lb, len(df)) if i > 0
        )
    )
    break_down = (
        float(c["close"]) < orb_low - buffer and    # still below ORB now
        any(
            df["close"].iloc[i-1] >= orb_low - buffer and df["close"].iloc[i] < orb_low - buffer
            for i in range(len(df) - lb, len(df)) if i > 0
        )
    )

    # Don't fire if we're already far from ORB (late-day re-test, not a clean break)
    too_far_up   = float(c["close"]) > orb_high + 2.5 * orb_size
    too_far_down = float(c["close"]) < orb_low  - 2.5 * orb_size

    if break_up and ema_bull and vol_ok and rsi < 75 and not too_far_up:
        conf = _score_confidence(df, 0.76, "long")
        return StrategySignal(
            "ORB Breakout", "long", close,
            tp_price   = round(close + 3.0 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"Break above ORB ${orb_high:.2f}", "EMA9>21",
                          f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}", f"ORB size ${orb_size:.2f}"],
            exit_hints = [f"Price falls back below ORB ${orb_high:.2f}",
                          "MACD crosses down with vol > 1.5x",
                          "RSI > 80"],
        )

    if break_down and ema_bear and vol_ok and rsi > 25 and not too_far_down:
        conf = _score_confidence(df, 0.76, "short")
        return StrategySignal(
            "ORB Breakout", "short", close,
            tp_price   = round(close - 3.0 * atr, 2),
            sl_price   = round(close + 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"Break below ORB ${orb_low:.2f}", "EMA9<21",
                          f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}", f"ORB size ${orb_size:.2f}"],
            exit_hints = [f"Price reclaims ORB ${orb_low:.2f}",
                          "MACD crosses up with vol > 1.5x",
                          "RSI < 25"],
        )

    return None


# ── Strategy 8: Oversold Dip Buy ─────────────────────────────────────────────

def strategy_oversold_dip(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    RSI < 25 panic flush while the medium-term trend is still up (EMA21 > EMA50).
    Buy the fear, ride the snap-back. After first 30 min only.

    Walk-forward validated on 17 months of SPY 1-min (128k bars, Alpaca):
    test edge +0.37, WR 49%, ~1.5 fires/day. Held-out edge BEAT training
    edge (decay -50%) — the opposite of overfit.
    """
    if not _has(df, "close","rsi","ema21","ema50","atr"):
        return None
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    ema21 = _get(df, "ema21"); ema50 = _get(df, "ema50")
    # Explicit None checks — a full-panic flush gives RSI exactly 0.0, which is
    # falsy and would wrongly fail an all([...]) check right when we want to fire
    if any(v is None for v in (close, atr, rsi, ema21, ema50)) or atr <= 0:
        return None

    mins = _get(df, "mins_since_open") or 999
    if mins < 30: return None

    if rsi < 25 and ema21 > ema50:
        conf = _score_confidence(df, 0.78, "long")
        return StrategySignal(
            "Oversold Dip Buy", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"RSI {rsi:.0f} panic flush", "EMA21>50 uptrend intact"],
            exit_hints = ["RSI recovers above 60",
                          "EMA21 crosses below EMA50 — uptrend broken"],
        )
    return None


# ── Strategy 9: Opening Drive ─────────────────────────────────────────────────

def strategy_opening_drive(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    First 30 minutes: trade WITH the directional money flow (+DI vs -DI).
    The single most stable setup in the 17-month sweep — long side decayed
    only 1% between training and held-out data.

    NOTE: agent skips scans before 9:45 ET (Vishy rule), so the live
    window is 9:45-10:00. Fires nearly every day — will usually consume
    one of the 2 daily trade slots.

    Backtest: long E=+0.31 WR=50%, short E=+0.27 WR=48%, ~1.1 fires/day
    """
    if not _has(df, "close","plus_di","minus_di","rsi","atr","ema9","ema21","volume"):
        return None
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    pdi   = _get(df, "plus_di"); mdi = _get(df, "minus_di")
    ema9  = _get(df, "ema9");    ema21 = _get(df, "ema21")
    if not all([close, atr, rsi, ema9, ema21]) or atr <= 0: return None
    if pdi is None or mdi is None: return None

    mins = _get(df, "mins_since_open")
    if mins is None or mins >= 30: return None

    # DI must be decisively separated — marginal crosses whipsaw at the open.
    # Jun 15 9:45 fired SHORT on -DI 25 vs +DI 17 while EMA9>EMA21 (bullish),
    # exited for $0.00 in seconds. The EMA-coherence requirement blocks that.
    di_sep = abs(pdi - mdi)
    if di_sep < 8: return None

    # Volume floor — the worst opens fired on 0.3x dead volume.
    vol_ma = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    vol_r  = float(df["volume"].iloc[-1]) / vol_ma if vol_ma > 0 else 1.0
    if vol_r < 0.8: return None

    if pdi > mdi and ema9 > ema21:           # DI and EMA both bullish
        conf = _score_confidence(df, 0.74, "long")
        return StrategySignal(
            "Opening Drive", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"+DI {pdi:.0f} > -DI {mdi:.0f} (sep {di_sep:.0f})",
                          "EMA9>21 confirms", f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price breaks below VWAP",
                          "-DI crosses above +DI", "RSI > 75"],
        )

    if mdi > pdi and ema9 < ema21:           # DI and EMA both bearish
        conf = _score_confidence(df, 0.74, "short")
        return StrategySignal(
            "Opening Drive", "short", close,
            tp_price   = round(close - 2.5 * atr, 2),
            sl_price   = round(close + 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"-DI {mdi:.0f} > +DI {pdi:.0f} (sep {di_sep:.0f})",
                          "EMA9<21 confirms", f"Vol {vol_r:.1f}x", f"RSI {rsi:.0f}"],
            exit_hints = ["Price reclaims VWAP",
                          "+DI crosses above -DI", "RSI < 25"],
        )
    return None


# ── Strategy 10: Lunch VWAP Hold ──────────────────────────────────────────────

def strategy_lunch_vwap_hold(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Lunch hour (12:00-12:59 ET): price holding above session VWAP means
    buyers stayed in control through the midday chop — institutions add
    into the afternoon. Long only: the short variant tested weak (+0.12)
    and is deliberately excluded.

    Backtest: E=+0.31  WR=49%  ~1.2 fires/day  (held-out beat training, decay -25%)
    """
    if not _has(df, "close","vwap","rsi","atr"):
        return None
    close = _get(df, "close"); atr = _get(df, "atr"); rsi = _get(df, "rsi")
    vwap  = _get(df, "vwap")
    hour_et = _get(df, "hour_et")
    if not all([close, atr, rsi, vwap]) or atr <= 0: return None
    if hour_et is None or hour_et != 12: return None

    # Require price a clear 0.25xATR ABOVE vwap, not just touching it. On Jun 13
    # and Jun 15 this fired 4x each day while price chopped right on top of vwap
    # (25% live win rate). The buffer demands genuine separation.
    if close > vwap + 0.25 * atr:
        conf = _score_confidence(df, 0.72, "long")
        return StrategySignal(
            "Lunch VWAP Hold", "long", close,
            tp_price   = round(close + 2.5 * atr, 2),
            sl_price   = round(close - 1.0 * atr, 2),
            confidence = conf,
            reasons    = [f"${close - vwap:.2f} above VWAP through lunch",
                          f"RSI {rsi:.0f}"],
            exit_hints = ["Price breaks below VWAP",
                          "RSI > 75", "EMA9 crosses below EMA21"],
        )
    return None


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "ORB Breakout":     strategy_orb,                # clean level break with volume
    "Power Hour Dip":   strategy_power_hour_dip,    # E=+1.00 WR 74%  3pm only
    "Keltner Bounce":   strategy_keltner_bounce,     # E=+0.73 WR 64%  ranging
    "Momentum Flip":    strategy_momentum_flip,      # E=+0.55 WR 62%  daily signal
    "Trend Breakout":   strategy_trend_breakout,     # E=+0.45 WR 54%  trending
    "VWAP Reclaim":     strategy_vwap_reclaim,       # E=+0.39 WR 52%  vol confirm
    "BB+ADX Reversal":  strategy_bb_adx_reversal,    # E=+0.23 WR 55%  ranging
    # v5 — walk-forward validated on 17mo SPY 1-min (backtest_explorer.py, Jun 2026)
    "Oversold Dip Buy": strategy_oversold_dip,       # E=+0.37 WR 49%  panic dip in uptrend
    "Opening Drive":    strategy_opening_drive,      # E=+0.31 WR 50%  9:45-10:00 DI direction
    "Lunch VWAP Hold":  strategy_lunch_vwap_hold,    # E=+0.31 WR 49%  12pm VWAP hold, long only
}


def run_all_strategies(
    df: pd.DataFrame,
    enabled: list = None,
    symbol: str = "",
    notify: bool = True,
) -> List[StrategySignal]:
    fns = {k: v for k, v in ALL_STRATEGIES.items()
           if enabled is None or k in enabled}
    regime = market_regime(df)
    signals = []
    for name, fn in fns.items():
        try:
            sig = fn(df)
            if sig is not None:
                # Regime overlay: penalize counter-regime conviction, annotate fit.
                fit = regime_fit(sig.strategy, regime)
                if fit == "poor":
                    sig.confidence = round(max(0.55, sig.confidence - 0.12), 2)
                    sig.reasons.append(f"⚠️ {regime} regime — counter to setup")
                elif fit == "good":
                    sig.reasons.append(f"✓ {regime} regime aligned")
                signals.append(sig)
                if notify and symbol:
                    try:
                        _get_tg().send_signal(
                            symbol=symbol, strategy=sig.strategy,
                            signal="BUY" if sig.side == "long" else "SELL",
                            price=sig.entry_price,
                            details="  |  ".join(sig.reasons) +
                                    f"\nConfidence: {sig.conf_label}",
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    return signals
