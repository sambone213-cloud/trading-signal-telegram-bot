"""Every signal flagged today — no trade caps, live pipeline (_compute_indicators).
Dedup matches live bot: same (strategy, side) suppressed within 2xATR of last fire.
"""
import warnings; warnings.filterwarnings("ignore")
import datetime
import pandas as pd
import yfinance as yf
from trading_alert_agent import _compute_indicators
from trade_strategies import run_all_strategies

hist = yf.Ticker("SPY").history(period="5d", interval="1m", auto_adjust=False)
hist.index = hist.index.tz_convert("UTC")
full = pd.DataFrame({
    "datetime": hist.index,
    "open": hist["Open"].values, "high": hist["High"].values,
    "low": hist["Low"].values, "close": hist["Close"].values,
    "volume": hist["Volume"].values,
}).sort_values("datetime").reset_index(drop=True)

sim_date = full["datetime"].dt.tz_convert("America/New_York").dt.date.iloc[-1]
print(f"Raw signal scan for {sim_date} (live pipeline, no trade caps)\n")

day_idx = full.index[full["datetime"].dt.tz_convert("America/New_York").dt.date == sim_date]
start = day_idx[0]

last_fire = {}   # (strategy, side) -> price
flags = []

for i in range(start, day_idx[-1] + 1):
    window = _compute_indicators(full.iloc[:i + 1])
    sigs = run_all_strategies(window, notify=False)
    if not sigs:
        continue
    price = float(window["close"].iloc[-1])
    atr   = float(window["atr"].iloc[-1]) if pd.notna(window["atr"].iloc[-1]) else 0.5
    et    = window["datetime"].iloc[-1].tz_convert("America/New_York").strftime("%H:%M")
    for s in sigs:
        key = (s.strategy, s.side)
        if key in last_fire and abs(price - last_fire[key]) < 2 * atr:
            continue
        last_fire[key] = price
        flags.append((et, s.strategy, s.side, price, s.confidence, " | ".join(s.reasons)))

print(f"{'time':>5}  {'strategy':<18} {'side':<5} {'price':>8}  conf  reasons")
print("-" * 110)
for et, strat, side, price, conf, reasons in flags:
    print(f"{et:>5}  {strat:<18} {side:<5} {price:>8.2f}  {conf:.2f}  {reasons}")
print(f"\nTotal raw flags today: {len(flags)}")
by = {}
for f in flags:
    by[f[1]] = by.get(f[1], 0) + 1
for k, v in sorted(by.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
