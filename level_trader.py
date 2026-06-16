"""
level_trader.py — break / retest / continuation level plays
───────────────────────────────────────────────────────────
Trades the key levels the briefing already computes (prior H/L/C, premarket
H/L, swing highs/lows). Mirrors how Sam reads a chart by hand:

    "price breaks 760, retests 757 and holds, then aim for 763"

The monitor runs a small state machine per level across scans:
    idle → broke (price clears level by a buffer and extends away)
         → retested (price pulls back to the level but holds)
         → CONFIRMED (price resumes in the break direction) → ALERT

On confirmation it fires a play: entry confirmed, target = the NEXT level in
that direction (or a measured move if none), stop = back through the level.
"""

import datetime


# ── Ladder helpers ────────────────────────────────────────────────────────────

def build_ladder(levels, price):
    """Return (resistance_above, support_below) sorted outward from price."""
    above = sorted([l for l in levels if l > price])
    below = sorted([l for l in levels if l < price], reverse=True)
    return above, below


def next_target(levels, level, direction):
    """Next level beyond `level` in the given direction, or None."""
    if direction == "up":
        higher = sorted([l for l in levels if l > level + 0.01])
        return higher[0] if higher else None
    lower = sorted([l for l in levels if l < level - 0.01], reverse=True)
    return lower[0] if lower else None


def briefing_plays(levels, price, atr, max_each=2):
    """If-break-then-target lines for the market-open briefing."""
    if not levels or atr <= 0 or price <= 0:
        return []
    above, below = build_ladder(levels, price)
    lines = []
    for r in above[:max_each]:
        tgt = next_target(levels, r, "up")
        tstr = f"${tgt:.2f}" if tgt else f"${r + 2 * atr:.2f} (+2ATR)"
        lines.append(f"▲ Break ${r:.2f} + hold retest → target {tstr}")
    for s in below[:max_each]:
        tgt = next_target(levels, s, "down")
        tstr = f"${tgt:.2f}" if tgt else f"${s - 2 * atr:.2f} (-2ATR)"
        lines.append(f"▼ Lose ${s:.2f} + reject retest → target {tstr}")
    return lines


# ── Live break/retest/continuation monitor ────────────────────────────────────

class LevelMonitor:
    """
    Stateful across scans. Call update() each scan with the latest price.
    Returns a list of formatted alert strings when a level play confirms.
    One alert per level per day (state resets at the day boundary).
    """

    def __init__(self):
        self._state = {}   # (symbol, level) -> dict
        self._day = None

    def _roll_day(self, dt):
        d = dt.date()
        if self._day != d:
            self._state.clear()
            self._day = d

    def update(self, symbol, levels, price, atr, dt, regime="?"):
        self._roll_day(dt)
        if not levels or atr <= 0 or price <= 0:
            return []

        brk     = max(0.10 * atr, 0.04)   # decisive break buffer
        push    = 0.50 * atr              # must extend this far past the level
        retest  = 0.20 * atr              # pullback comes back within this of level
        confirm = 0.40 * atr              # resume this far past level to confirm
        alerts  = []

        for lvl in levels:
            key = (symbol, round(lvl, 2))
            st  = self._state.get(key, {"phase": "idle", "done": False})
            if st.get("done"):
                continue
            phase = st["phase"]

            if phase == "idle":
                if price > lvl + brk:
                    st = {"phase": "broke_up", "ext": price, "retested": False, "done": False}
                elif price < lvl - brk:
                    st = {"phase": "broke_dn", "ext": price, "retested": False, "done": False}

            elif phase == "broke_up":
                st["ext"] = max(st["ext"], price)
                if price < lvl - brk:                                   # failed break
                    st = {"phase": "idle", "done": False}
                elif (not st["retested"] and st["ext"] >= lvl + push
                      and price <= lvl + retest):                       # pulled back to retest
                    st["retested"] = True
                elif st["retested"] and price >= lvl + confirm:         # resumed up → confirm
                    tgt = next_target(levels, lvl, "up")
                    alerts.append(self._fmt(symbol, "up", lvl, price, tgt, atr, regime))
                    st["done"] = True

            elif phase == "broke_dn":
                st["ext"] = min(st["ext"], price)
                if price > lvl + brk:                                   # failed break
                    st = {"phase": "idle", "done": False}
                elif (not st["retested"] and st["ext"] <= lvl - push
                      and price >= lvl - retest):                       # pulled back to retest
                    st["retested"] = True
                elif st["retested"] and price <= lvl - confirm:         # resumed down → confirm
                    tgt = next_target(levels, lvl, "down")
                    alerts.append(self._fmt(symbol, "down", lvl, price, tgt, atr, regime))
                    st["done"] = True

            self._state[key] = st

        return alerts

    @staticmethod
    def _fmt(symbol, direction, lvl, price, tgt, atr, regime):
        if direction == "up":
            tstr = f"${tgt:.2f}" if tgt else f"${lvl + 2 * atr:.2f} (+2ATR)"
            stop = lvl - 0.4 * atr
            arrow, side = "▲", "CALL"
            line = f"${lvl:.2f} broke → retested → holding"
        else:
            tstr = f"${tgt:.2f}" if tgt else f"${lvl - 2 * atr:.2f} (-2ATR)"
            stop = lvl + 0.4 * atr
            arrow, side = "▼", "PUT"
            line = f"${lvl:.2f} lost → retested → rejecting"
        reg = ("✓ " + regime) if regime in ("TREND_UP", "TREND_DOWN") else \
              ("⚠️ " + regime + " (fakeout risk)" if regime == "CHOP" else regime)
        return (
            f"📐 <b>LEVEL PLAY — {symbol}</b>  {arrow} {side}\n"
            f"──────────────────────\n"
            f"{line}\n"
            f"Now ${price:.2f}  →  🎯 Target {tstr}\n"
            f"🛑 Stop ${stop:.2f} (back through level)\n"
            f"Regime: {reg}"
        )
