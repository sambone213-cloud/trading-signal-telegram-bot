"""Replay today taking EVERY signal — no 2-trade cap, no single-position rule.
Each signal runs as its own independent trade: exits on TP/SL touch (intrabar),
strategy exit condition, or end of day. Dedup matches live bot (2xATR per strategy/side).
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
from trading_alert_agent import _compute_indicators
from trade_strategies import run_all_strategies, check_exit_conditions

hist = yf.Ticker("SPY").history(period="5d", interval="1m", auto_adjust=False)
hist.index = hist.index.tz_convert("UTC")
full = pd.DataFrame({
    "datetime": hist.index,
    "open": hist["Open"].values, "high": hist["High"].values,
    "low": hist["Low"].values, "close": hist["Close"].values,
    "volume": hist["Volume"].values,
}).sort_values("datetime").reset_index(drop=True)

et_dates = full["datetime"].dt.tz_convert("America/New_York").dt.date
sim_date = et_dates.iloc[-1]
day_idx = full.index[et_dates == sim_date]
start, end = day_idx[0], day_idx[-1]

open_trades = []   # dicts: strategy, side, entry_price, entry_et, tp, sl, conf, entry_bar
closed = []
last_fire = {}

# Precompute indicator windows once per bar is expensive; reuse the window df
for i in range(start, end + 1):
    window = _compute_indicators(full.iloc[:i + 1])
    bar = window.iloc[-1]
    price = float(bar["close"]); hi = float(bar["high"]); lo = float(bar["low"])
    et = window["datetime"].iloc[-1].tz_convert("America/New_York").strftime("%H:%M")

    # 1. Manage open trades (skip the entry bar itself)
    still_open = []
    for t in open_trades:
        if i == t["entry_bar"]:
            still_open.append(t); continue
        exit_px, cause = None, None
        if t["side"] == "long":
            if lo <= t["sl"]:   exit_px, cause = t["sl"], "SL hit"
            elif hi >= t["tp"]: exit_px, cause = t["tp"], "TP hit"
        else:
            if hi >= t["sl"]:   exit_px, cause = t["sl"], "SL hit"
            elif lo <= t["tp"]: exit_px, cause = t["tp"], "TP hit"
        if exit_px is None:
            hint = check_exit_conditions(window, t["side"], t["strategy"], i - t["entry_bar"])
            if hint:
                exit_px, cause = price, hint
        if exit_px is None:
            still_open.append(t)
        else:
            pnl = (exit_px - t["entry_price"]) if t["side"] == "long" else (t["entry_price"] - exit_px)
            closed.append({**t, "exit_et": et, "exit_price": exit_px,
                           "cause": cause, "pnl": pnl})
    open_trades = still_open

    # 2. New signals — every one becomes a trade
    sigs = run_all_strategies(window, notify=False)
    atr = float(bar["atr"]) if pd.notna(bar["atr"]) else 0.5
    for s in sigs:
        key = (s.strategy, s.side)
        if key in last_fire and abs(price - last_fire[key]) < 2 * atr:
            continue
        last_fire[key] = price
        open_trades.append({
            "strategy": s.strategy, "side": s.side, "entry_price": price,
            "entry_et": et, "tp": s.tp_price, "sl": s.sl_price,
            "conf": s.confidence, "entry_bar": i,
        })

# EOD close anything open
eod_px = float(full["close"].iloc[end])
for t in open_trades:
    pnl = (eod_px - t["entry_price"]) if t["side"] == "long" else (t["entry_price"] - eod_px)
    closed.append({**t, "exit_et": "EOD", "exit_price": eod_px,
                   "cause": "end of day", "pnl": pnl})

closed.sort(key=lambda t: t["entry_et"])
print(f"UNCAPPED REPLAY — {sim_date}  (every signal taken, independent trades)\n")
print(f"{'entry':>5} {'exit':>5}  {'strategy':<18} {'side':<5} {'conf':>4}  "
      f"{'in':>8} {'out':>8}  {'P&L $':>7}  cause")
print("-" * 105)
total = wins = 0
for t in closed:
    total += t["pnl"]; wins += t["pnl"] > 0
    print(f"{t['entry_et']:>5} {t['exit_et']:>5}  {t['strategy']:<18} {t['side']:<5} "
          f"{t['conf']:.2f}  {t['entry_price']:>8.2f} {t['exit_price']:>8.2f}  "
          f"{t['pnl']:>+7.2f}  {t['cause']}")
print("-" * 105)
print(f"Trades: {len(closed)}  |  Wins: {wins}/{len(closed)} "
      f"({wins/len(closed)*100:.0f}%)  |  Net SPY points: {total:+.2f}")

print("\nPer strategy:")
by = {}
for t in closed:
    d = by.setdefault(t["strategy"], {"n": 0, "w": 0, "pnl": 0.0})
    d["n"] += 1; d["w"] += t["pnl"] > 0; d["pnl"] += t["pnl"]
for k, d in sorted(by.items(), key=lambda x: -x[1]["pnl"]):
    print(f"  {k:<18} {d['n']} trade(s)  {d['w']}/{d['n']} wins  {d['pnl']:+.2f} pts")

print("\nHIGH confidence only (>= 0.80):")
hc = [t for t in closed if t["conf"] >= 0.80]
if hc:
    hw = sum(1 for t in hc if t["pnl"] > 0)
    print(f"  {len(hc)} trade(s)  {hw}/{len(hc)} wins  {sum(t['pnl'] for t in hc):+.2f} pts")
