"""
position_manager.py
────────────────────
Tracks the bot's "virtual" position state so it never fires a BUY and then a
SELL (or vice versa) within minutes of each other — the exact problem from
11:44–11:48 AM and the 1:03/1:47 PM signals on 6/10.

Usage in trading_alert_agent.py:

    from position_manager import PositionManager
    pm = PositionManager(lockout_minutes=15, max_trades_per_day=2)

    ...inside scan loop, after run_all_strategies() returns signals...

    for sig in signals:
        ok, reason = pm.evaluate(sig, bar_time)
        if not ok:
            print(f"[SUPPRESSED] {sig.strategy} {sig.side} — {reason}")
            continue
        # send to Telegram as normal
        pm.register(sig, bar_time)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple


@dataclass
class PositionManager:
    lockout_minutes: int = 15        # min time before an OPPOSITE signal is allowed
    max_trades_per_day: int = 2

    _active_side: Optional[str] = field(default=None, init=False)   # 'long' | 'short' | None
    _active_strategy: Optional[str] = field(default=None, init=False)
    _entry_time: Optional[datetime] = field(default=None, init=False)
    _entry_price: Optional[float] = field(default=None, init=False)
    _trades_today: int = field(default=0, init=False)
    _current_day: Optional[str] = field(default=None, init=False)

    # ── Day rollover ───────────────────────────────────────────────────────
    def _check_new_day(self, bar_time: datetime):
        day = bar_time.strftime("%Y-%m-%d")
        if self._current_day != day:
            self._current_day = day
            self._trades_today = 0
            self._active_side = None
            self._active_strategy = None
            self._entry_time = None
            self._entry_price = None

    # ── Main gate ─────────────────────────────────────────────────────────
    def evaluate(self, signal, bar_time: datetime) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Call this BEFORE sending any Telegram signal.
        """
        self._check_new_day(bar_time)

        if self._trades_today >= self.max_trades_per_day:
            return False, f"max {self.max_trades_per_day} trades/day reached"

        # No open position — anything goes (subject to max trades)
        if self._active_side is None:
            return True, "no active position"

        # Same direction as current position — duplicate, suppress
        if signal.side == self._active_side:
            return False, (
                f"already in a {self._active_side} position "
                f"(opened by {self._active_strategy} @ {self._entry_price})"
            )

        # Opposite direction — only allow if lockout window has passed
        elapsed_min = (bar_time - self._entry_time).total_seconds() / 60.0
        if elapsed_min < self.lockout_minutes:
            return False, (
                f"opposite signal only {elapsed_min:.1f}m after "
                f"{self._active_strategy} {self._active_side} entry "
                f"(lockout = {self.lockout_minutes}m)"
            )

        # Lockout passed — allow as a position flip (counts as a new trade)
        return True, "lockout passed — treated as exit + reverse"

    # ── State updates ────────────────────────────────────────────────────
    def register(self, signal, bar_time: datetime):
        """Call AFTER a signal passes evaluate() and is sent to Telegram."""
        self._check_new_day(bar_time)
        flipped = self._active_side is not None and signal.side != self._active_side
        self._active_side = signal.side
        self._active_strategy = signal.strategy
        self._entry_time = bar_time
        self._entry_price = signal.entry_price
        self._trades_today += 1
        return "flip" if flipped else "new"

    def force_close(self, bar_time: datetime):
        """Call when an exit (manual or exit_manager) closes the position
        WITHOUT consuming a new trade slot — keeps trades_today count intact
        but frees up the lockout for the next directional signal."""
        self._check_new_day(bar_time)
        self._active_side = None
        self._active_strategy = None
        self._entry_time = None
        self._entry_price = None

    # ── Status ───────────────────────────────────────────────────────────
    @property
    def status(self) -> str:
        if self._active_side is None:
            return f"FLAT  |  trades today: {self._trades_today}/{self.max_trades_per_day}"
        return (
            f"{self._active_side.upper()} via {self._active_strategy} "
            f"@ {self._entry_price}  |  trades today: "
            f"{self._trades_today}/{self.max_trades_per_day}"
        )
