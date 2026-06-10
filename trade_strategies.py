"""
trade_strategies.py  — v2
─────────────────────────
Added two new strategies vs v1:
  • VWMA Cross     — volume-weighted MA cross with ADX trend confirmation
  • BB+ADX Reversal — BB mean reversion ONLY in ranging markets (ADX < 25)

Backtest ranking by edge (expectancy per trade):
  1. BB+ADX Reversal  E=+0.228  WR 27%  ◄ NEW #1
  2. VWAP Reclaim     E=+0.155  WR 34%
  3. Trend Short      E=+0.130  WR 38%
  4. VWMA Cross       E=+0.109  WR 35%  ◄ NEW #4
  5. EMA Cross        E=+0.072  WR 33%
  6. BB/RSI (old)     E=+0.053  WR 11%  ← replaced by BB+ADX
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


# ─── Signal model ─────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    strategy:    str
    side:        str          # 'long' | 'short'
    entry_price: float
    tp_price:    float
    sl_price:    float
    confidence:  float
    reasons:     list = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        tp_dist = abs(self.tp_price - self.entry_price)
        sl_dist = abs(self.sl_price - self.entry_price)
        return round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, col: str, idx: int = -1):
    try:
        return float(df[col].iloc[idx])
    except Exception:
        return None


def _has(df: pd.DataFrame, *cols) -> bool:
    return not df.empty and len(df) >= 30 and all(c in df.columns for c in cols)


# ─── Indicator: ADX ───────────────────────────────────────────────────────────
# trading_alert_agent.py must add these columns before calling strategies.
# Add to _compute_indicators() in trading_alert_agent.py:
#
#   plus_dm  = (df["high"].diff()).clip(lower=0)
#   minus_dm = (-df["low"].diff()).clip(lower=0)
#   overlap  = (df["high"].diff() > 0) & (-df["low"].diff() > 0)
#   plus_dm[overlap & (df["high"].diff() <  -df["low"].diff())] = 0
#   minus_dm[overlap & (df["high"].diff() > -df["low"].diff())] = 0
#   atr14    = tr.rolling(14).mean()
#   df["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
#   df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
#   dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
#   df["adx"] = dx.rolling(14).mean()
#
#   vw9        = (df["close"] * df["volume"]).rolling(9).sum()  / df["volume"].rolling(9).sum()
#   vw21       = (df["close"] * df["volume"]).rolling(21).sum() / df["volume"].rolling(21).sum()
#   df["vwma9"]  = vw9
#   df["vwma21"] = vw21


# ─── Strategy 1: EMA Golden / Death Cross ─────────────────────────────────────

def strategy_ema_cross(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Trend-following: EMA9 crosses EMA21 in the direction of EMA50 trend.
    Long  : EMA9 crosses above EMA21  AND  close > EMA50  AND  RSI 35–65
    Short : EMA9 crosses below EMA21  AND  close < EMA50  AND  RSI 35–65
    TP: 2.5×ATR   SL: 1.0×ATR   R:R ≈ 2.5
    """
    if not _has(df, "ema9", "ema21", "ema50", "rsi", "atr", "close"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr   = _get(df, "atr")
    close = _get(df, "close")
    rsi   = _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None
    ema_up   = p["ema9"] <= p["ema21"] and c["ema9"] > c["ema21"]
    ema_down = p["ema9"] >= p["ema21"] and c["ema9"] < c["ema21"]
    if ema_up and close > _get(df, "ema50") and 35 <= rsi <= 65 and atr >= 0.05:
        return StrategySignal(
            "EMA Cross", "long", close,
            round(close + 2.5 * atr, 4), round(close - 1.0 * atr, 4), 0.70,
            reasons=["EMA9×21↑", "↑EMA50 trend", f"RSI {rsi:.0f}"],
        )
    if ema_down and close < _get(df, "ema50") and 35 <= rsi <= 65 and atr >= 0.05:
        return StrategySignal(
            "EMA Cross", "short", close,
            round(close - 2.5 * atr, 4), round(close + 1.0 * atr, 4), 0.70,
            reasons=["EMA9×21↓", "↓EMA50 trend", f"RSI {rsi:.0f}"],
        )
    return None


# ─── Strategy 2: MACD Momentum ────────────────────────────────────────────────

def strategy_macd_momentum(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Momentum: MACD crosses signal line with expanding histogram.
    Long  : MACD×↑  AND  MACD < 0  AND  RSI < 58  AND  EMA9 > EMA21
    Short : MACD×↓  AND  MACD > 0  AND  RSI > 42  AND  EMA9 < EMA21
    TP: 2.5×ATR   SL: 1.0×ATR
    """
    if not _has(df, "macd", "macd_signal", "macd_hist", "rsi", "atr", "close", "ema9", "ema21"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None
    ema_bull = float(c["ema9"]) > float(c["ema21"])
    ema_bear = float(c["ema9"]) < float(c["ema21"])
    mac_up   = p["macd"] <= p["macd_signal"] and c["macd"] > c["macd_signal"]
    mac_down = p["macd"] >= p["macd_signal"] and c["macd"] < c["macd_signal"]
    hist_exp = abs(float(c["macd_hist"])) > abs(float(p["macd_hist"]))
    if mac_up and float(c["macd"]) < 0 and rsi < 58 and hist_exp and ema_bull:
        return StrategySignal(
            "MACD Momentum", "long", close,
            round(close + 2.5 * atr, 4), round(close - 1.0 * atr, 4), 0.68,
            reasons=["MACD×↑", "Below 0", f"RSI {rsi:.0f}", "Hist↑", "EMA9>21"],
        )
    if mac_down and float(c["macd"]) > 0 and rsi > 42 and hist_exp and ema_bear:
        return StrategySignal(
            "MACD Momentum", "short", close,
            round(close - 2.5 * atr, 4), round(close + 1.0 * atr, 4), 0.68,
            reasons=["MACD×↓", "Above 0", f"RSI {rsi:.0f}", "Hist↑", "EMA9<21"],
        )
    return None


# ─── Strategy 3: BB Mean Reversion (original — kept for reference) ────────────

def strategy_bb_mean_reversion(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Original BB Reversal — no ADX filter.
    NOTE: 11% win rate in backtest. Superseded by strategy_bb_adx_reversal.
    Kept in codebase but NOT in ALL_STRATEGIES registry.
    """
    if not _has(df, "bb_upper", "bb_lower", "bb_mid", "rsi", "atr", "close", "ema9", "ema21"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    rsi_prev = float(p["rsi"]) if "rsi" in p.index else rsi
    bb_lo  = _get(df, "bb_lower")
    bb_hi  = _get(df, "bb_upper")
    bb_mid = _get(df, "bb_mid")
    if not all([atr, close, rsi, bb_lo, bb_hi, bb_mid]) or atr <= 0:
        return None
    ema_bull = float(c["ema9"]) > float(c["ema21"])
    ema_bear = float(c["ema9"]) < float(c["ema21"])
    two_low  = float(c["close"]) <= float(c["bb_lower"]) and float(p["close"]) <= float(p["bb_lower"])
    two_high = float(c["close"]) >= float(c["bb_upper"]) and float(p["close"]) >= float(p["bb_upper"])
    rsi_up, rsi_dn = rsi > rsi_prev, rsi < rsi_prev
    if two_low and rsi < 32 and rsi_up and ema_bull:
        return StrategySignal("BB Reversal", "long", close,
                              round(bb_mid, 4), round(close - 0.6 * atr, 4), 0.68,
                              reasons=["2× below BB_lo", f"RSI {rsi:.0f}↑", "EMA9>21"])
    if two_high and rsi > 68 and rsi_dn and ema_bear:
        return StrategySignal("BB Reversal", "short", close,
                              round(bb_mid, 4), round(close + 0.6 * atr, 4), 0.68,
                              reasons=["2× above BB_hi", f"RSI {rsi:.0f}↓", "EMA9<21"])
    one_high = float(c["close"]) >= float(c["bb_upper"])
    one_low  = float(c["close"]) <= float(c["bb_lower"])
    if one_high and rsi >= 78 and rsi_dn:
        return StrategySignal("RSI Exhaustion", "short", close,
                              round(bb_mid, 4), round(close + 0.5 * atr, 4), 0.70,
                              reasons=["RSI Exhausted", f"RSI {rsi:.0f}↓", "1× above BB_hi"])
    if one_low and rsi <= 25 and rsi_up:
        return StrategySignal("RSI Exhaustion", "long", close,
                              round(bb_mid, 4), round(close - 0.5 * atr, 4), 0.70,
                              reasons=["RSI Exhausted", f"RSI {rsi:.0f}↑", "1× below BB_lo"])
    return None


# ─── Strategy 4: VWAP Reclaim ─────────────────────────────────────────────────

def strategy_vwap_reclaim(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Price crosses VWAP with volume surge — institutional flow signal.
    Long  : prior close < VWAP, current close > VWAP  AND  Vol > 1.4× avg
    Short : prior close > VWAP, current close < VWAP  AND  Vol > 1.4× avg
    TP: 2.5×ATR   SL: 0.8×ATR   R:R ≈ 3.1
    Backtest edge: +0.155 pts/trade  (ranked #2)
    """
    if not _has(df, "vwap", "close", "volume", "atr", "rsi"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None
    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.4
    cross_up   = float(p["close"]) <= float(p["vwap"]) and float(c["close"]) > float(c["vwap"])
    cross_down = float(p["close"]) >= float(p["vwap"]) and float(c["close"]) < float(c["vwap"])
    if cross_up and vol_ok and rsi < 68:
        return StrategySignal(
            "VWAP Reclaim", "long", close,
            round(close + 2.5 * atr, 4), round(close - 0.8 * atr, 4), 0.65,
            reasons=["VWAP×↑", "Vol surge", f"RSI {rsi:.0f}"],
        )
    if cross_down and vol_ok and rsi > 32:
        return StrategySignal(
            "VWAP Reclaim", "short", close,
            round(close - 2.5 * atr, 4), round(close + 0.8 * atr, 4), 0.65,
            reasons=["VWAP×↓", "Vol surge", f"RSI {rsi:.0f}"],
        )
    return None


# ─── Strategy 5: Trend Continuation Short ─────────────────────────────────────

def strategy_trend_continuation_short(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Dead-cat bounce failure in a downtrend — selling resumes.
    Short : EMA9 < EMA21  AND  RSI bounced then rolling back  AND  MACD hist more negative  AND  vol spike
    TP: 2.0×ATR   SL: 0.8×ATR
    Backtest edge: +0.130 pts/trade  (ranked #3)
    """
    if not _has(df, "ema9", "ema21", "macd_hist", "rsi", "atr", "close", "volume"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None
    ema_bear      = float(c["ema9"]) < float(c["ema21"])
    rsi_was_up    = float(p["rsi"]) >= 48
    rsi_rolling   = rsi < float(p["rsi"]) and rsi < 62
    hist_resuming = float(c["macd_hist"]) < float(p["macd_hist"])
    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.3
    if ema_bear and rsi_was_up and rsi_rolling and hist_resuming and vol_ok:
        return StrategySignal(
            "Trend Short", "short", close,
            round(close - 2.0 * atr, 4), round(close + 0.8 * atr, 4), 0.67,
            reasons=["EMA9<21", f"RSI {float(p['rsi']):.0f}→{rsi:.0f}↓", "Hist↓", "Vol↑"],
        )
    return None


# ─── Strategy 6: VWMA Cross  ★ NEW ───────────────────────────────────────────

def strategy_vwma_cross(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Volume-Weighted MA cross — like EMA Cross but weighted by volume.
    High-volume crossovers carry more conviction than thin-bar crossovers.
    Requires ADX > 20 to confirm a trend before entering.

    Long  : VWMA9 crosses above VWMA21  AND  ADX > 20  AND  close > EMA50  AND  RSI 40–65
    Short : VWMA9 crosses below VWMA21  AND  ADX > 20  AND  close < EMA50  AND  RSI 35–60
    TP: 2.5×ATR   SL: 1.0×ATR
    Backtest edge: +0.109 pts/trade  WR 35%  (ranked #4)

    Requires in _compute_indicators():
        vw9  = (close * volume).rolling(9).sum()  / volume.rolling(9).sum()
        vw21 = (close * volume).rolling(21).sum() / volume.rolling(21).sum()
        df["vwma9"], df["vwma21"] = vw9, vw21
        df["adx"] = ... (see ADX block in this file header)
    """
    if not _has(df, "vwma9", "vwma21", "adx", "rsi", "atr", "close", "ema50"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    adx = _get(df, "adx")
    if not all([atr, close, rsi, adx]) or atr <= 0:
        return None
    if adx < 20:
        return None  # flat market — skip
    cross_up   = float(p["vwma9"]) <= float(p["vwma21"]) and float(c["vwma9"]) > float(c["vwma21"])
    cross_down = float(p["vwma9"]) >= float(p["vwma21"]) and float(c["vwma9"]) < float(c["vwma21"])
    ema50 = _get(df, "ema50")
    if cross_up and close > ema50 and 40 <= rsi <= 65 and atr >= 0.05:
        return StrategySignal(
            "VWMA Cross", "long", close,
            round(close + 2.5 * atr, 4), round(close - 1.0 * atr, 4), 0.75,
            reasons=["VWMA9×21↑", f"ADX {adx:.0f}", f"RSI {rsi:.0f}"],
        )
    if cross_down and close < ema50 and 35 <= rsi <= 60 and atr >= 0.05:
        return StrategySignal(
            "VWMA Cross", "short", close,
            round(close - 2.5 * atr, 4), round(close + 1.0 * atr, 4), 0.75,
            reasons=["VWMA9×21↓", f"ADX {adx:.0f}", f"RSI {rsi:.0f}"],
        )
    return None


# ─── Strategy 7: BB + ADX Mean Reversion  ★ NEW ──────────────────────────────

def strategy_bb_adx_reversal(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    BB Mean Reversion ONLY in ranging markets (ADX < 25).
    This fixes the original BB Reversal's 11% win rate — it was firing in
    trending markets where mean reversion fails. ADX < 25 confirms chop/range.

    Long  : 2 bars at/below BB lower  AND  RSI < 35  AND  RSI turning up  AND  ADX < 25
    Short : 2 bars at/above BB upper  AND  RSI > 65  AND  RSI turning down  AND  ADX < 25
    TP: BB midline   SL: 0.6×ATR
    Backtest edge: +0.228 pts/trade  WR 27%  ★ RANKED #1 overall
    """
    if not _has(df, "bb_upper", "bb_lower", "bb_mid", "rsi", "atr", "close", "adx"):
        return None
    c, p  = df.iloc[-1], df.iloc[-2]
    atr, close, rsi = _get(df, "atr"), _get(df, "close"), _get(df, "rsi")
    rsi_prev = float(p["rsi"]) if "rsi" in p.index else rsi
    bb_lo  = _get(df, "bb_lower")
    bb_hi  = _get(df, "bb_upper")
    bb_mid = _get(df, "bb_mid")
    adx    = _get(df, "adx")
    if not all([atr, close, rsi, bb_lo, bb_hi, bb_mid, adx]) or atr <= 0:
        return None
    if adx >= 25:
        return None  # trending market — do NOT fade the move
    two_low  = float(c["close"]) <= float(c["bb_lower"]) and float(p["close"]) <= float(p["bb_lower"])
    two_high = float(c["close"]) >= float(c["bb_upper"]) and float(p["close"]) >= float(p["bb_upper"])
    rsi_up, rsi_dn = rsi > rsi_prev, rsi < rsi_prev
    if two_low and rsi < 35 and rsi_up:
        return StrategySignal(
            "BB+ADX Reversal", "long", close,
            round(bb_mid, 4), round(close - 0.6 * atr, 4), 0.75,
            reasons=["2×BB_lo", f"RSI {rsi:.0f}↑", f"ADX {adx:.0f} ranging"],
        )
    if two_high and rsi > 65 and rsi_dn:
        return StrategySignal(
            "BB+ADX Reversal", "short", close,
            round(bb_mid, 4), round(close + 0.6 * atr, 4), 0.75,
            reasons=["2×BB_hi", f"RSI {rsi:.0f}↓", f"ADX {adx:.0f} ranging"],
        )
    return None


# ─── Strategy 8: ORB (unchanged from v1) ──────────────────────────────────────

def strategy_orb(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Opening Range Breakout — first 15-min NY session range.
    Long  : close > ORB_high  AND  vol > avg  (after 13:45 UTC)
    Short : close < ORB_low   AND  vol > avg
    TP: ORB_high + 2×range   SL: ORB midpoint   R:R ≈ 2
    """
    if not _has(df, "close", "high", "low", "datetime", "volume", "atr"):
        return None
    df2 = df.copy()
    df2["_h"] = df2["datetime"].dt.hour + df2["datetime"].dt.minute / 60.0
    _today = df2["datetime"].iloc[-1].strftime("%Y-%m-%d")
    orb_bars = df2[
        (df2["datetime"].dt.strftime("%Y-%m-%d") == _today) &
        (df2["_h"] >= 13.5) & (df2["_h"] < 13.75)
    ]
    if orb_bars.empty:
        return None
    orb_high  = float(orb_bars["high"].max())
    orb_low   = float(orb_bars["low"].min())
    orb_range = orb_high - orb_low
    orb_mid   = (orb_high + orb_low) / 2
    if orb_range < 1e-6:
        return None
    c = df2.iloc[-1]
    if float(c["_h"]) < 13.75:
        return None
    close = float(c["close"])
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    if not all([atr, rsi]) or atr <= 0:
        return None
    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.2
    prev_close = float(df2.iloc[-2]["close"]) if len(df2) >= 2 else close
    if close > orb_high and prev_close <= orb_high and vol_ok:
        return StrategySignal(
            "ORB", "long", close,
            round(orb_high + 2.0 * orb_range, 4), round(orb_mid, 4), 0.68,
            reasons=["ORB Break↑", f"Range ${orb_range:.2f}", "Vol↑"],
        )
    if close < orb_low and prev_close >= orb_low and vol_ok:
        return StrategySignal(
            "ORB", "short", close,
            round(orb_low - 2.0 * orb_range, 4), round(orb_mid, 4), 0.68,
            reasons=["ORB Break↓", f"Range ${orb_range:.2f}", "Vol↑"],
        )
    return None


# ─── Strategy 9: Gap & Go (unchanged from v1) ─────────────────────────────────

def strategy_gap_and_go(df: pd.DataFrame) -> Optional[StrategySignal]:
    if not _has(df, "close", "open", "datetime", "volume", "atr", "rsi"):
        return None
    df2 = df.copy()
    df2["_h"] = df2["datetime"].dt.hour + df2["datetime"].dt.minute / 60.0
    _today = df2["datetime"].iloc[-1].strftime("%Y-%m-%d")
    prior_bars = df2[df2["datetime"].dt.strftime("%Y-%m-%d") < _today]
    if prior_bars.empty:
        return None
    prior_close = float(prior_bars["close"].iloc[-1])
    today_bars  = df2[df2["datetime"].dt.strftime("%Y-%m-%d") == _today]
    if today_bars.empty:
        return None
    today_open = float(today_bars["open"].iloc[0])
    gap_pct    = (today_open - prior_close) / prior_close
    c     = df2.iloc[-1]
    close = float(c["close"])
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    if not all([atr, rsi]) or atr <= 0:
        return None
    if not (13.5 <= float(c["_h"]) < 14.0):
        return None
    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.5
    if gap_pct <= -0.005 and close < today_open and vol_ok and rsi < 55:
        return StrategySignal(
            "Gap & Go", "short", close,
            round(close - 2.0 * atr, 4), round(close + 0.8 * atr, 4), 0.70,
            reasons=[f"Gap↓ {gap_pct*100:.1f}%", "Continuation↓", "Vol↑"],
        )
    if gap_pct >= 0.005 and close > today_open and vol_ok and rsi > 45:
        return StrategySignal(
            "Gap & Go", "long", close,
            round(close + 2.0 * atr, 4), round(close - 0.8 * atr, 4), 0.70,
            reasons=[f"Gap↑ {gap_pct*100:.1f}%", "Continuation↑", "Vol↑"],
        )
    return None


# ─── Registry ─────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    # Ranked by backtest edge (expectancy per trade):
    "BB+ADX Reversal": strategy_bb_adx_reversal,    # E=+0.228  ★ #1
    "VWAP Reclaim":    strategy_vwap_reclaim,        # E=+0.155     #2
    "Trend Short":     strategy_trend_continuation_short,  # E=+0.130  #3
    "VWMA Cross":      strategy_vwma_cross,          # E=+0.109  ★ #4
    "EMA Cross":       strategy_ema_cross,           # E=+0.072     #5
    "MACD Momentum":   strategy_macd_momentum,       # E=+0.068     #6
    "ORB":             strategy_orb,
    "Gap & Go":        strategy_gap_and_go,
    # "BB Reversal":   strategy_bb_mean_reversion,   # RETIRED — 11% WR, replaced by BB+ADX
}

_CRYPTO_INCOMPATIBLE = {"ORB"}

def _is_crypto_symbol(symbol: str) -> bool:
    s = (symbol or "").upper()
    return "/" in s or any(s.startswith(c) for c in
                           ("BTC","ETH","SOL","ADA","DOT","MATIC","AVAX","LINK","UNI","XRP","LTC","DOGE"))


def run_all_strategies(
    df: pd.DataFrame,
    enabled: list = None,
    symbol: str = "",
    notify: bool = True,
) -> List[StrategySignal]:
    crypto = _is_crypto_symbol(symbol)
    fns = {k: v for k, v in ALL_STRATEGIES.items()
           if (enabled is None or k in enabled)
           and not (crypto and k in _CRYPTO_INCOMPATIBLE)}
    signals = []
    for name, fn in fns.items():
        try:
            sig = fn(df)
            if sig is not None:
                signals.append(sig)
                if notify and symbol:
                    try:
                        tg_signal = "BUY" if sig.side == "long" else "SELL"
                        details   = "  ·  ".join(sig.reasons) if sig.reasons else ""
                        _get_tg().send_signal(
                            symbol=symbol, strategy=sig.strategy,
                            signal=tg_signal, price=sig.entry_price,
                            details=details,
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    return signals
