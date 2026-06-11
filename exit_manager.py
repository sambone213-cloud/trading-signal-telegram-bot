"""
exit_manager.py
───────────────
Tiered exit system for 0DTE options positions.

Tier 1 — Hard floor (fires immediately, non-negotiable)
    Exit 50% of position at 1.5× premium paid → locks profit, rest is free

Tier 2 — Dynamic trail on remaining 50%
    Exit when RSI crosses back against direction
    Exit when price recrosses VWAP against direction
    Exit when EMA9 crosses back against direction

Tier 3 — Runner condition (let it ride toward 4-5×)
    Only if: ADX > 25 AND price moved > 1.5×ATR AND no reversal signals 3 scans

Usage in trading_alert_agent.py scan loop:
    from exit_manager import ExitManager
    em = ExitManager(notifier=_get_tg())
    ...
    em.register_entry(symbol, sig, entry_premium, bar_df)
    ...
    em.evaluate(symbol, current_premium, bar_df)  # call every scan
"""

import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict
from trade_strategies import StrategySignal

try:
    from zoneinfo import ZoneInfo
    _ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    _ET_ZONE = None


def _now_et() -> datetime.datetime:
    """Eastern time regardless of server timezone (Railway runs UTC)."""
    utc = datetime.datetime.now(datetime.timezone.utc)
    if _ET_ZONE is not None:
        return utc.astimezone(_ET_ZONE).replace(tzinfo=None)
    return utc.replace(tzinfo=None) - datetime.timedelta(hours=4)


@dataclass
class OpenPosition:
    symbol:          str
    strategy:        str
    side:            str            # 'long' | 'short'
    entry_price:     float          # underlying price at entry
    entry_premium:   float          # option premium paid
    entry_time:      datetime.datetime
    entry_df_rsi:    float          # RSI at entry
    entry_df_vwap:   float          # VWAP at entry
    entry_df_ema9:   float          # EMA9 at entry

    # State
    tier1_done:      bool  = False  # half exited at 1.5×
    clean_scans:     int   = 0      # consecutive scans with no reversal signal
    peak_premium:    float = 0.0    # highest premium seen since entry


class ExitManager:
    def __init__(self, notifier=None):
        self._positions: Dict[str, OpenPosition] = {}
        self._tg = notifier

    def register_entry(
        self,
        symbol: str,
        sig: StrategySignal,
        entry_premium: float,
        df,
    ):
        """Call when a signal fires and you enter a position."""
        try:
            rsi  = float(df["rsi"].iloc[-1])
            vwap = float(df["vwap"].iloc[-1]) if "vwap" in df.columns else 0.0
            ema9 = float(df["ema9"].iloc[-1])
        except Exception:
            rsi, vwap, ema9 = 50.0, 0.0, sig.entry_price

        self._positions[symbol] = OpenPosition(
            symbol=symbol,
            strategy=sig.strategy,
            side=sig.side,
            entry_price=sig.entry_price,
            entry_premium=entry_premium,
            entry_time=_now_et(),
            entry_df_rsi=rsi,
            entry_df_vwap=vwap,
            entry_df_ema9=ema9,
            peak_premium=entry_premium,
        )
        print(f"  [ExitMgr] Tracking {symbol} {sig.side} @ ${entry_premium:.2f} premium")

    def close_position(self, symbol: str):
        """Call when manually closed or stopped out."""
        self._positions.pop(symbol, None)

    def evaluate(self, symbol: str, current_premium: float, df) -> Optional[str]:
        """
        Call every scan. Returns exit reason string if exit triggered, else None.
        Fires Telegram alert on exit signal.

        NOTE: current_premium is ignored for P&L — ATR×0.5 is not a real option price.
        Exit timing is driven purely by underlying price action (RSI, VWAP, EMA crosses).
        P&L display shows the underlying move against the entry price instead.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        try:
            rsi   = float(df["rsi"].iloc[-1])
            vwap  = float(df["vwap"].iloc[-1]) if "vwap" in df.columns else 0.0
            ema9  = float(df["ema9"].iloc[-1])
            ema21 = float(df["ema21"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])
            close = float(df["close"].iloc[-1])
            adx   = float(df["adx"].iloc[-1]) if "adx" in df.columns else 20.0
        except Exception:
            return None

        # Underlying move since entry — this is what actually matters
        raw_move  = close - pos.entry_price
        # For a short, a move DOWN is profitable; for a long, a move UP is profitable
        direction = -1 if pos.side == "short" else 1
        underlying_pnl = direction * raw_move   # positive = in your favour

        # ── Tier 1: underlying moved 1×ATR in your favour → alert to scale out ─
        if not pos.tier1_done and underlying_pnl >= 1.0 * atr:
            pos.tier1_done = True
            self._send(
                symbol, pos, close, underlying_pnl,
                f"🟡 TIER 1 — Underlying moved ${underlying_pnl:+.2f} in your favour\n"
                f"Consider selling 50% of your option position"
            )
            return f"TIER1 at +{underlying_pnl:.2f}"

        # ── Tier 2: reversal signal → exit remaining ─────────────────────────
        reversal = self._reversal_check(pos, rsi, vwap, ema9, ema21, close)
        if reversal:
            pos.clean_scans = 0
            runner = (
                adx > 25 and
                underlying_pnl > 1.5 * atr and
                pos.tier1_done
            )
            if runner:
                self._send(symbol, pos, close, underlying_pnl,
                           f"🚀 RUNNER — {reversal} but ADX {adx:.0f} strong\n"
                           f"Holding runner — monitor closely")
                return f"RUNNER: {reversal}"
            else:
                return self._fire_exit(symbol, pos, close, underlying_pnl,
                                       f"🔴 EXIT SIGNAL — {reversal}", tier=2)
        else:
            pos.clean_scans += 1

        return None

    def _reversal_check(self, pos, rsi, vwap, ema9, ema21, close) -> Optional[str]:
        """Returns reversal reason string if exit condition met, else None."""
        if pos.side == "long":
            if rsi > 72:
                return f"RSI overbought {rsi:.0f}"
            if vwap > 0 and close < vwap and pos.entry_df_vwap > 0 and pos.entry_price > vwap:
                return "Price crossed below VWAP"
            if ema9 < ema21:
                return "EMA9 crossed below EMA21"
        else:  # short
            if rsi < 28:
                return f"RSI oversold {rsi:.0f}"
            if vwap > 0 and close > vwap and pos.entry_df_vwap > 0 and pos.entry_price < vwap:
                return "Price crossed above VWAP"
            if ema9 > ema21:
                return "EMA9 crossed above EMA21"
        return None

    def _fire_exit(self, symbol, pos, close, underlying_pnl, reason, tier) -> str:
        self._send(symbol, pos, close, underlying_pnl, reason)
        self._positions.pop(symbol, None)
        return reason

    def _send(self, symbol, pos, close, underlying_pnl, message):
        now_et    = _now_et()
        held      = int((now_et - pos.entry_time).total_seconds() / 60)
        move_str  = f"${underlying_pnl:+.2f} {'in your favour' if underlying_pnl > 0 else 'against you'}"
        text = (
            f"{message}\n"
            f"──────────────────────\n"
            f"📐 {pos.strategy}  {pos.side.upper()}\n"
            f"💲 Entry: ${pos.entry_price:.2f}  →  Now: ${close:.2f}\n"
            f"📊 Underlying move: {move_str}  |  Held: {held}m\n"
            f"🕐 {now_et.strftime('%H:%M ET')}"
        )
        print(f"\n{message} — {symbol} underlying {move_str}")
        if self._tg:
            try:
                self._tg.send_raw(text)
            except Exception:
                pass
