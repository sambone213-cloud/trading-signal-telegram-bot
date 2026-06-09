"""
Trade strategies for the QuantDesk paper-trading simulation.

Each strategy function accepts a full OHLCV + indicators DataFrame and
returns a StrategySignal if entry conditions are met on the latest bar,
or None if conditions are not satisfied.

Risk:Reward targets per strategy:
  EMA Cross        2.5 : 1   (trend-following)
  MACD Momentum    2.5 : 1   (momentum)
  BB Reversal      ~2  : 1   (mean-reversion, TP = BB midline)
  VWAP Reclaim     2.5 : 1   (intraday momentum)
  ORB Breakout     2   : 1   (opening range, first-30-min NY)
"""

from dataclasses import dataclass, field
from typing import Optional, List
import pandas as pd
import numpy as np

# Telegram notifications (optional — no-ops when credentials not set)
try:
    from telegram_notify import get_notifier as _get_tg
except ImportError:
    def _get_tg():  # type: ignore
        class _Noop:
            def send_signal(self, *a, **kw): pass
        return _Noop()


# ─── Signal model ────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    strategy:    str
    side:        str          # 'long' | 'short'
    entry_price: float
    tp_price:    float
    sl_price:    float
    confidence:  float        # 0.0–1.0
    reasons:     list = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        tp_dist = abs(self.tp_price - self.entry_price)
        sl_dist = abs(self.sl_price - self.entry_price)
        return round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, col: str, idx: int = -1):
    try:
        return float(df[col].iloc[idx])
    except Exception:
        return None


def _has(df: pd.DataFrame, *cols) -> bool:
    return not df.empty and len(df) >= 3 and all(c in df.columns for c in cols)


# ─── Strategy 1: EMA Golden / Death Cross ────────────────────────────────────

def strategy_ema_cross(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Trend-following: EMA9 crosses EMA21 in the direction of EMA50 trend.

    Long  : EMA9 crosses above EMA21  AND  close > EMA50  AND  RSI 35–65
    Short : EMA9 crosses below EMA21  AND  close < EMA50  AND  RSI 35–65
    TP: 2.5 × ATR   SL: 1.0 × ATR   R:R ≈ 2.5
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
            strategy="EMA Cross", side="long", entry_price=close,
            tp_price=round(close + 2.5 * atr, 4),
            sl_price=round(close - 1.0 * atr, 4),
            confidence=0.70,
            reasons=["EMA9×21↑", "↑EMA50 trend", f"RSI {rsi:.0f}"],
        )
    if ema_down and close < _get(df, "ema50") and 35 <= rsi <= 65 and atr >= 0.05:
        return StrategySignal(
            strategy="EMA Cross", side="short", entry_price=close,
            tp_price=round(close - 2.5 * atr, 4),
            sl_price=round(close + 1.0 * atr, 4),
            confidence=0.70,
            reasons=["EMA9×21↓", "↓EMA50 trend", f"RSI {rsi:.0f}"],
        )
    return None


# ─── Strategy 2: MACD Momentum ───────────────────────────────────────────────

def strategy_macd_momentum(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Momentum: MACD crosses signal line while histogram momentum is building.

    Long  : MACD crosses above signal  AND  MACD < 0 (early / fresh cross)  AND  RSI < 58
    Short : MACD crosses below signal  AND  MACD > 0  AND  RSI > 42
    TP: 2.5 × ATR   SL: 1.0 × ATR   R:R ≈ 2.5
    """
    if not _has(df, "macd", "macd_signal", "macd_hist", "rsi", "atr", "close", "ema9", "ema21"):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    atr   = _get(df, "atr")
    close = _get(df, "close")
    rsi   = _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None

    ema_bull = float(c["ema9"]) > float(c["ema21"])
    ema_bear = float(c["ema9"]) < float(c["ema21"])

    mac_up   = p["macd"] <= p["macd_signal"] and c["macd"] > c["macd_signal"]
    mac_down = p["macd"] >= p["macd_signal"] and c["macd"] < c["macd_signal"]
    hist_expanding = abs(float(c["macd_hist"])) > abs(float(p["macd_hist"]))

    if mac_up and float(c["macd"]) < 0 and rsi < 58 and hist_expanding and ema_bull:
        return StrategySignal(
            strategy="MACD Momentum", side="long", entry_price=close,
            tp_price=round(close + 2.5 * atr, 4),
            sl_price=round(close - 1.0 * atr, 4),
            confidence=0.68,
            reasons=["MACD×↑", "Below 0", f"RSI {rsi:.0f}", "Hist↑", "EMA9>21"],
        )
    if mac_down and float(c["macd"]) > 0 and rsi > 42 and hist_expanding and ema_bear:
        return StrategySignal(
            strategy="MACD Momentum", side="short", entry_price=close,
            tp_price=round(close - 2.5 * atr, 4),
            sl_price=round(close + 1.0 * atr, 4),
            confidence=0.68,
            reasons=["MACD×↓", "Above 0", f"RSI {rsi:.0f}", "Hist↑", "EMA9<21"],
        )
    return None


# ─── Strategy 3: Bollinger Band Mean Reversion ───────────────────────────────

def strategy_bb_mean_reversion(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Mean reversion: price outside BB with RSI extreme AND trend confirmation.

    Long  : close ≤ BB_lower  AND  RSI < 32  AND  RSI turning up  AND  EMA9 > EMA21
    Short : close ≥ BB_upper  AND  RSI > 68  AND  RSI turning down  AND  EMA9 < EMA21
    TP: BB midline   SL: 0.6 × ATR   R:R ≈ 1.5–2.5
    """
    if not _has(df, "bb_upper", "bb_lower", "bb_mid", "rsi", "atr", "close", "ema9", "ema21"):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    atr    = _get(df, "atr")
    close  = _get(df, "close")
    rsi    = _get(df, "rsi")
    rsi_prev = float(p["rsi"]) if "rsi" in p.index else rsi
    bb_lo  = _get(df, "bb_lower")
    bb_hi  = _get(df, "bb_upper")
    bb_mid = _get(df, "bb_mid")
    if not all([atr, close, rsi, bb_lo, bb_hi, bb_mid]) or atr <= 0:
        return None

    ema_bull = float(c["ema9"]) > float(c["ema21"])  # trend filter
    ema_bear = float(c["ema9"]) < float(c["ema21"])

    two_bars_low  = float(c["close"]) <= float(c["bb_lower"]) and float(p["close"]) <= float(p["bb_lower"])
    two_bars_high = float(c["close"]) >= float(c["bb_upper"]) and float(p["close"]) >= float(p["bb_upper"])

    rsi_turning_up   = rsi > rsi_prev   # RSI must be rising for long
    rsi_turning_down = rsi < rsi_prev   # RSI must be falling for short

    if two_bars_low and rsi < 32 and rsi_turning_up and ema_bull:
        return StrategySignal(
            strategy="BB Reversal", side="long", entry_price=close,
            tp_price=round(bb_mid, 4),
            sl_price=round(close - 0.6 * atr, 4),
            confidence=0.68,
            reasons=["2× below BB_lo", f"RSI {rsi:.0f}↑", "EMA9>21"],
        )
    if two_bars_high and rsi > 68 and rsi_turning_down and ema_bear:
        return StrategySignal(
            strategy="BB Reversal", side="short", entry_price=close,
            tp_price=round(bb_mid, 4),
            sl_price=round(close + 0.6 * atr, 4),
            confidence=0.68,
            reasons=["2× above BB_hi", f"RSI {rsi:.0f}↓", "EMA9<21"],
        )

    # RSI exhaustion — fires regardless of EMA trend when severely overbought/oversold
    # Only needs 1 bar outside BB since BB tracks price upward on strong trends
    one_bar_high = float(c["close"]) >= float(c["bb_upper"])
    one_bar_low  = float(c["close"]) <= float(c["bb_lower"])
    rsi_exhausted_short = one_bar_high and rsi >= 78 and rsi_turning_down
    rsi_exhausted_long  = one_bar_low  and rsi <= 25 and rsi_turning_up

    if rsi_exhausted_short:
        return StrategySignal(
            strategy="RSI Exhaustion", side="short", entry_price=close,
            tp_price=round(bb_mid, 4),
            sl_price=round(close + 0.5 * atr, 4),
            confidence=0.70,
            reasons=["RSI Exhausted", f"RSI {rsi:.0f}↓", "2× above BB_hi"],
        )
    if rsi_exhausted_long:
        return StrategySignal(
            strategy="RSI Exhaustion", side="long", entry_price=close,
            tp_price=round(bb_mid, 4),
            sl_price=round(close - 0.5 * atr, 4),
            confidence=0.70,
            reasons=["RSI Exhausted", f"RSI {rsi:.0f}↑", "2× below BB_lo"],
        )
    return None


# ─── Strategy 4: VWAP Reclaim ────────────────────────────────────────────────

def strategy_vwap_reclaim(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Intraday: price crosses VWAP with a volume surge — institutional flow signal.

    Long  : prior close < VWAP, current close > VWAP  AND  Vol > 1.4× 20-bar avg
    Short : prior close > VWAP, current close < VWAP  AND  Vol > 1.4× 20-bar avg
    TP: 2.5 × ATR   SL: 0.8 × ATR   R:R ≈ 3.1
    """
    if not _has(df, "vwap", "close", "volume", "atr", "rsi"):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    atr   = _get(df, "atr")
    close = _get(df, "close")
    rsi   = _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None

    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.4

    cross_up   = float(p["close"]) <= float(p["vwap"]) and float(c["close"]) > float(c["vwap"])
    cross_down = float(p["close"]) >= float(p["vwap"]) and float(c["close"]) < float(c["vwap"])

    if cross_up and vol_ok and rsi < 68:
        return StrategySignal(
            strategy="VWAP Reclaim", side="long", entry_price=close,
            tp_price=round(close + 2.5 * atr, 4),
            sl_price=round(close - 0.8 * atr, 4),
            confidence=0.65,
            reasons=["VWAP×↑", "Vol surge", f"RSI {rsi:.0f}"],
        )
    if cross_down and vol_ok and rsi > 32:
        return StrategySignal(
            strategy="VWAP Reclaim", side="short", entry_price=close,
            tp_price=round(close - 2.5 * atr, 4),
            sl_price=round(close + 0.8 * atr, 4),
            confidence=0.65,
            reasons=["VWAP×↓", "Vol surge", f"RSI {rsi:.0f}"],
        )
    return None


# ─── Strategy 5: Opening Range Breakout (ORB) ────────────────────────────────

def strategy_orb(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    ORB: trade breakouts beyond the first 15-min NY session range.
    NY open = 09:30 ET = 13:30 UTC.  ORB window: 13:30–13:45 UTC.

    Long  : close > ORB_high  AND  vol > avg  (after 13:45 UTC)
    Short : close < ORB_low   AND  vol > avg
    TP: ORB_high + 2 × ORB_range   SL: ORB midpoint   R:R ≈ 2
    """
    if not _has(df, "close", "high", "low", "datetime", "volume", "atr"):
        return None

    df2       = df.copy()
    df2["_h"] = df2["datetime"].dt.hour + df2["datetime"].dt.minute / 60.0
    # Only use TODAY's opening range — never bleed historical days into the range calc
    _today_date = df2["datetime"].iloc[-1].strftime("%Y-%m-%d")
    orb_bars  = df2[
        (df2["datetime"].dt.strftime("%Y-%m-%d") == _today_date) &
        (df2["_h"] >= 13.5) &
        (df2["_h"] < 13.75)   # 13:45 UTC = 9:45 ET (15-min ORB)
    ]
    if orb_bars.empty:
        return None

    orb_high  = float(orb_bars["high"].max())
    orb_low   = float(orb_bars["low"].min())
    orb_range = orb_high - orb_low
    orb_mid   = (orb_high + orb_low) / 2
    if orb_range < 1e-6:
        return None

    c     = df2.iloc[-1]
    # Must be PAST the ORB window
    if float(c["_h"]) < 13.75:
        return None

    close  = float(c["close"])
    atr    = _get(df, "atr")
    rsi    = _get(df, "rsi")
    if not all([atr, rsi]) or atr <= 0:
        return None

    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.2

    # Only fire on the FIRST bar that breaks the range (prev bar was inside)
    prev_close = float(df2.iloc[-2]["close"]) if len(df2) >= 2 else close
    first_break_up   = close > orb_high and prev_close <= orb_high
    first_break_down = close < orb_low  and prev_close >= orb_low

    if first_break_up and vol_ok:
        return StrategySignal(
            strategy="ORB", side="long", entry_price=close,
            tp_price=round(orb_high + 2.0 * orb_range, 4),
            sl_price=round(orb_mid, 4),
            confidence=0.68,
            reasons=["ORB Break↑", f"Range ${orb_range:.2f}", "Vol↑"],
        )
    if first_break_down and vol_ok:
        return StrategySignal(
            strategy="ORB", side="short", entry_price=close,
            tp_price=round(orb_low - 2.0 * orb_range, 4),
            sl_price=round(orb_mid, 4),
            confidence=0.68,
            reasons=["ORB Break↓", f"Range ${orb_range:.2f}", "Vol↑"],
        )
    return None


# ─── Strategy 6: Trend Continuation Short ────────────────────────────────────

def strategy_trend_continuation_short(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Catches continuation moves in an established downtrend.
    Fires when a dead-cat bounce fails and selling resumes.

    Short : EMA9 < EMA21 (downtrend)
            AND RSI bounced to 48-68 then rolling back down
            AND MACD histogram turning more negative (momentum resuming down)
            AND volume spike on the reversal bar
    TP: 2.0 × ATR   SL: 0.8 × ATR   R:R ≈ 2.5
    """
    if not _has(df, "ema9", "ema21", "macd_hist", "rsi", "atr", "close", "volume"):
        return None

    c, p  = df.iloc[-1], df.iloc[-2]
    atr   = _get(df, "atr")
    close = _get(df, "close")
    rsi   = _get(df, "rsi")
    if not all([atr, close, rsi]) or atr <= 0:
        return None

    ema_bear      = float(c["ema9"]) < float(c["ema21"])
    rsi_was_up    = float(p["rsi"]) >= 48                        # bounce got RSI up
    rsi_rolling   = rsi < float(p["rsi"]) and rsi < 62          # now rolling back down
    hist_resuming = float(c["macd_hist"]) < float(p["macd_hist"])  # histogram getting more negative

    vol_ma  = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok  = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.3

    if ema_bear and rsi_was_up and rsi_rolling and hist_resuming and vol_ok:
        return StrategySignal(
            strategy="Trend Short", side="short", entry_price=close,
            tp_price=round(close - 2.0 * atr, 4),
            sl_price=round(close + 0.8 * atr, 4),
            confidence=0.67,
            reasons=["EMA9<21", f"RSI {float(p['rsi']):.0f}→{rsi:.0f}↓", "Hist↓", "Vol↑"],
        )
    return None


# ─── Strategy 7: Gap & Go ────────────────────────────────────────────────────

def strategy_gap_and_go(df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Detects a significant gap at open and trades the continuation.
    Gap down ≥ 0.5% → short (sell the gap continuation)
    Gap up   ≥ 0.5% → long  (buy the gap continuation)
    Only fires in the first 30 minutes after open (9:30–10:00 ET).
    Requires price moving IN the gap direction with volume.
    """
    if not _has(df, "close", "open", "datetime", "volume", "atr", "rsi"):
        return None

    df2         = df.copy()
    df2["_h"]   = df2["datetime"].dt.hour + df2["datetime"].dt.minute / 60.0
    _today_date = df2["datetime"].iloc[-1].strftime("%Y-%m-%d")

    # Find prior day's close — last bar before today
    prior_bars = df2[df2["datetime"].dt.strftime("%Y-%m-%d") < _today_date]
    if prior_bars.empty:
        return None
    prior_close = float(prior_bars["close"].iloc[-1])

    # Find today's open — first bar of the day
    today_bars = df2[df2["datetime"].dt.strftime("%Y-%m-%d") == _today_date]
    if today_bars.empty:
        return None
    today_open = float(today_bars["open"].iloc[0])

    gap_pct = (today_open - prior_close) / prior_close

    c     = df2.iloc[-1]
    close = float(c["close"])
    atr   = _get(df, "atr")
    rsi   = _get(df, "rsi")
    if not all([atr, rsi]) or atr <= 0:
        return None

    # Only fire in first 30 minutes after open (13:30–14:00 UTC)
    current_h = float(c["_h"])
    if not (13.5 <= current_h < 14.0):
        return None

    vol_ma = df["volume"].rolling(20, min_periods=3).mean().iloc[-1]
    vol_ok = vol_ma > 0 and float(c["volume"]) > vol_ma * 1.5

    # Gap down ≥ 0.5% — short continuation
    if gap_pct <= -0.005 and close < today_open and vol_ok and rsi < 55:
        return StrategySignal(
            strategy="Gap & Go", side="short", entry_price=close,
            tp_price=round(close - 2.0 * atr, 4),
            sl_price=round(close + 0.8 * atr, 4),
            confidence=0.70,
            reasons=[f"Gap↓ {gap_pct*100:.1f}%", "Continuation↓", "Vol↑"],
        )

    # Gap up ≥ 0.5% — long continuation
    if gap_pct >= 0.005 and close > today_open and vol_ok and rsi > 45:
        return StrategySignal(
            strategy="Gap & Go", side="long", entry_price=close,
            tp_price=round(close + 2.0 * atr, 4),
            sl_price=round(close - 0.8 * atr, 4),
            confidence=0.70,
            reasons=[f"Gap↑ {gap_pct*100:.1f}%", "Continuation↑", "Vol↑"],
        )
    return None


# ─── Registry ────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "EMA Cross":      strategy_ema_cross,
    "MACD Momentum":  strategy_macd_momentum,
    "BB Reversal":    strategy_bb_mean_reversion,
    "VWAP Reclaim":   strategy_vwap_reclaim,
    "ORB":            strategy_orb,
    "Trend Short":    strategy_trend_continuation_short,
    "Gap & Go":       strategy_gap_and_go,
}


_CRYPTO_INCOMPATIBLE = {"ORB"}  # strategies that require equity market hours

def _is_crypto_symbol(symbol: str) -> bool:
    """Returns True for symbols like BTC/USD, ETH/USD, BTC-USD etc."""
    s = (symbol or "").upper()
    return "/" in s or any(s.startswith(c) for c in
                           ("BTC", "ETH", "SOL", "ADA", "DOT", "MATIC",
                            "AVAX", "LINK", "UNI", "XRP", "LTC", "DOGE"))

def run_all_strategies(df: pd.DataFrame,
                       enabled: list = None,
                       symbol: str = "",
                       notify: bool = True) -> List[StrategySignal]:
    """
    Run all (or a subset of) strategies on the latest bar.
    `enabled` is a list of strategy names; if None all are run.
    `symbol`  is used in Telegram notifications and crypto detection.
    `notify`  set False to suppress Telegram (e.g. during backtest replay).
    Returns all valid signals.
    """
    # Auto-exclude strategies that don't apply to 24/7 crypto markets
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
                # Telegram signal notification
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
