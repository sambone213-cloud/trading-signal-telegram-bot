"""
simulate_today.py
Runs SPY 1-min data through all 5 strategies bar-by-bar.
Fetches 5 days for indicator warmup, simulates the target date.
Usage: python simulate_today.py [YYYY-MM-DD]
       (defaults to today, falls back to last available day with signals)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, datetime
sys.path.insert(0, ".")
import pandas as pd, numpy as np
import yfinance as yf
from trade_strategies import run_all_strategies, check_exit_conditions

# ── Fetch 5 days for indicator warmup ────────────────────────────────────────
hist = yf.Ticker("SPY").history(period="5d", interval="1m", auto_adjust=False)
hist.index = hist.index.tz_convert("UTC")
full = pd.DataFrame({
    "datetime": hist.index,
    "open":   hist["Open"].values,
    "high":   hist["High"].values,
    "low":    hist["Low"].values,
    "close":  hist["Close"].values,
    "volume": hist["Volume"].values,
})
full["datetime"] = pd.to_datetime(full["datetime"], utc=True)
full = full.sort_values("datetime").reset_index(drop=True)

# ── Compute all indicators on full dataset ────────────────────────────────────
c, h, l, v = full["close"], full["high"], full["low"], full["volume"]

for span in [9, 21, 50]:
    full[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()

delta = c.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
full["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
full["macd"]        = ema12 - ema26
full["macd_signal"] = full["macd"].ewm(span=9, adjust=False).mean()
full["macd_hist"]   = full["macd"] - full["macd_signal"]

sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
full["bb_upper"] = sma20 + 2 * std20
full["bb_lower"] = sma20 - 2 * std20
full["bb_mid"]   = sma20

prev_c = c.shift(1)
tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
full["atr"]      = tr.rolling(14).mean()
full["kc_upper"] = sma20 + 2 * full["atr"]
full["kc_lower"] = sma20 - 2 * full["atr"]

plus_dm  = h.diff().clip(lower=0)
minus_dm = (-l.diff()).clip(lower=0)
atr14    = tr.rolling(14).mean()
full["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
full["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
dx = 100 * (full["plus_di"] - full["minus_di"]).abs() / \
     (full["plus_di"] + full["minus_di"]).replace(0, np.nan)
full["adx"] = dx.rolling(14).mean()

# VWAP resets each session
full["_date"] = full["datetime"].dt.date
full["vwap"]  = (c * v).groupby(full["_date"]).cumsum() / v.groupby(full["_date"]).cumsum()
full.drop(columns=["_date"], inplace=True)

full["vwma9"]    = (c * v).rolling(9).sum()  / v.rolling(9).sum()
full["vwma21"]   = (c * v).rolling(21).sum() / v.rolling(21).sum()
full["mom10"]    = c - c.shift(10)
full["roc10"]    = c.pct_change(10) * 100
full["vol_ratio"] = v / v.rolling(20, min_periods=5).mean()

lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
full["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
full["bb_pct"]  = (c - full["bb_lower"]) / (full["bb_upper"] - full["bb_lower"]).replace(0, np.nan)

full["hour_et"]         = full["datetime"].dt.hour - 4
full["minute_et"]       = full["datetime"].dt.minute
full["mins_since_open"] = (full["hour_et"] - 9) * 60 + full["minute_et"] - 30

full = full.dropna().reset_index(drop=True)

# ── Pick simulation date ──────────────────────────────────────────────────────
if len(sys.argv) > 1:
    sim_date = datetime.date.fromisoformat(sys.argv[1])
else:
    sim_date = full["datetime"].dt.date.iloc[-1]

# ── Confidence display ────────────────────────────────────────────────────────
def conf_label(conf):
    filled = int(round(conf * 5))
    bar    = ("*" * filled).ljust(5, "-")
    tier   = "HIGH" if conf >= 0.80 else ("MED " if conf >= 0.72 else "LOW ")
    return f"{tier}  [{bar}]  {conf*100:.0f}%"

# ── Bar-by-bar simulation on target date ─────────────────────────────────────
day_mask   = full["datetime"].dt.date == sim_date
day_start  = full.index[day_mask][0] if day_mask.any() else None

if day_start is None:
    print(f"No data for {sim_date}")
    sys.exit(1)

MAX_TRADES_DAY  = 2

events          = []
active          = None
last_entry_bar  = {}
trades_taken    = 0

for i in range(day_start, len(full)):
    if full["datetime"].iloc[i].date() != sim_date:
        break

    window = full.iloc[:i+1].copy()
    row    = full.iloc[i]
    et     = f"{int(row['hour_et']):02d}:{int(row['minute_et']):02d} ET"
    price  = float(row["close"])
    rsi    = float(row["rsi"])
    adx    = float(row["adx"])

    # Exit check
    if active:
        bars_held = i - active["entry_bar"]
        hint = check_exit_conditions(window, active["side"], active["strategy"], bars_held)
        if hint:
            pnl     = (price - active["entry_price"]) if active["side"] == "long" \
                      else (active["entry_price"] - price)
            pnl_pct = pnl / active["entry_price"] * 100
            events.append({
                "time": et, "type": "EXIT",
                "strategy": active["strategy"], "side": active["side"],
                "price": price, "entry": active["entry_price"],
                "pnl_pct": pnl_pct, "detail": hint,
                "conf": active["conf"], "bar": i,
            })
            active = None

    # Entry check — enforce max 2 trades/day (matches live position_manager rule)
    if active is None and trades_taken < MAX_TRADES_DAY:
        sigs = run_all_strategies(window, notify=False)
        for sig in sigs:
            if last_entry_bar.get(sig.strategy, -999) >= i - 15:
                continue
            events.append({
                "time": et, "type": "ENTRY",
                "strategy": sig.strategy, "side": sig.side,
                "price": price, "tp": sig.tp_price, "sl": sig.sl_price,
                "conf": sig.confidence, "reasons": sig.reasons,
                "exit_hints": sig.exit_hints,
                "rsi": rsi, "adx": adx, "bar": i,
            })
            last_entry_bar[sig.strategy] = i
            active = {
                "side": sig.side, "strategy": sig.strategy,
                "entry_price": price, "conf": sig.confidence,
                "entry_bar": i,
            }
            trades_taken += 1
            break

# ── Print report ──────────────────────────────────────────────────────────────
day_df  = full[full["datetime"].dt.date == sim_date]
open_p  = float(day_df["open"].iloc[0])
close_p = float(day_df["close"].iloc[-1])
day_pct = (close_p / open_p - 1) * 100
lo      = float(day_df["low"].min())
hi      = float(day_df["high"].max())

print()
print("=" * 64)
print(f"  SPY SIMULATION  {sim_date}")
print(f"  Open ${open_p:.2f}  Close ${close_p:.2f}  ({day_pct:+.2f}%)")
print(f"  Range ${lo:.2f} - ${hi:.2f}")
print("=" * 64)

if not events:
    print()
    last = day_df.iloc[-1]
    print("  No signals fired today with current strategy rules.")
    print()
    print(f"  ADX={last['adx']:.0f}  RSI={last['rsi']:.0f}  "
          f"EMA9 {'>' if last['ema9']>last['ema21'] else '<'} EMA21  "
          f"Price {'above' if last['close']>last['vwap'] else 'below'} VWAP")
    print()
    print("  Strategy gates:")
    print(f"    ORB Breakout     : break above/below 9:30-10am high/low + vol >1.5x")
    print(f"    Momentum Flip    : EMA9/21 cross + MACD cross + vol >2x  (fires daily)")
    print(f"    Keltner Bounce   : ADX<28 + plus_di>minus_di  (ranging, bullish)")
    print(f"    Power Hour Dip   : 3-4pm + EMA9>EMA21  (uptrend only)")
    print(f"    Trend Breakout   : ADX>22 + MACD cross + vol >1.4x")
    print(f"    VWAP Reclaim     : VWAP cross + 1.3x vol + EMA alignment")
    print(f"    BB+ADX Reversal  : ADX<25 + 2 bars outside BB")
else:
    total_pnl = 0.0
    for e in events:
        print()
        if e["type"] == "ENTRY":
            rr = abs(e["tp"] - e["price"]) / abs(e["price"] - e["sl"]) \
                 if abs(e["price"] - e["sl"]) > 0 else 0
            print(f"  [{e['time']}]  ** TELEGRAM ENTRY ALERT **")
            print(f"  " + "-" * 52)
            print(f"  Strategy   : {e['strategy']}")
            print(f"  Signal     : {e['side'].upper()}  SPY @ ${e['price']:.2f}")
            print(f"  TP / SL    : ${e['tp']:.2f} / ${e['sl']:.2f}  (R:R {rr:.1f}x)")
            print(f"  Confidence : {conf_label(e['conf'])}")
            print(f"  Conditions : {' | '.join(e['reasons'])}")
            for hint in e["exit_hints"]:
                print(f"  Watch exit : {hint}")
            print(f"  Context    : RSI {e['rsi']:.0f}  ADX {e['adx']:.0f}")
        else:
            result   = "WIN " if e["pnl_pct"] > 0 else "LOSS"
            total_pnl += e["pnl_pct"]
            print(f"  [{e['time']}]  ** TELEGRAM EXIT ALERT **   [{result}]")
            print(f"  " + "-" * 52)
            print(f"  Strategy   : {e['strategy']}  {e['side'].upper()}")
            print(f"  Trade      : ${e['entry']:.2f}  ->  ${e['price']:.2f}")
            print(f"  P&L        : {e['pnl_pct']:+.2f}%  (underlying SPY move)")
            print(f"  Exit cause : {e['detail']}")

    trades = [e for e in events if e["type"] == "EXIT"]
    if trades:
        wins = len([t for t in trades if t["pnl_pct"] > 0])
        print()
        print(f"  Closed: {len(trades)} trades  |  Wins: {wins}/{len(trades)}  |  "
              f"Total P&L: {total_pnl:+.2f}%")

    if active:
        pnl = (close_p - active["entry_price"]) if active["side"] == "long" \
              else (active["entry_price"] - close_p)
        pnl_pct = pnl / active["entry_price"] * 100
        print()
        print(f"  Open at EOD: {active['strategy']} {active['side'].upper()} "
              f"@ ${active['entry_price']:.2f}  ({pnl_pct:+.2f}% unrealized)")

print()
last = day_df.iloc[-1]
print("=" * 64)
print(f"  EOD: RSI {last['rsi']:.0f}  ADX {last['adx']:.0f}  "
      f"EMA9 {'>' if last['ema9']>last['ema21'] else '<'} EMA21  "
      f"Price {'above' if last['close']>last['vwap'] else 'below'} VWAP")
print("=" * 64)
