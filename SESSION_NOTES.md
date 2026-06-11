# QuantDesk Bot — Session Notes (Jun 11 2026)

## What This Bot Does
- Monitors SPY (and other symbols) live via yfinance 1-min bars
- Runs 7 strategies every 15 seconds, fires alerts to Telegram
- Targets 0DTE options, $0.30–$0.60 premium, max 2 trades/day
- Deployed on Railway, auto-restarts on failure

---

## Bugs Found & Fixed This Session

### Bug 1 — Keltner Channel columns missing (silent strategy failure)
**File:** `trading_alert_agent.py` → `_compute_indicators()`  
**Problem:** `kc_lower`, `kc_upper`, `kc_mid` were never computed. `strategy_keltner_bounce` checks for these columns via `_has()` and silently returned `None` every scan. Keltner Bounce was dead from day one.  
**Fix:** Added Keltner Channel computation to `_compute_indicators()`.

---

### Bug 2 — Volume always 0.0x (killed 4 of 7 strategies)
**File:** `trading_alert_agent.py` → `_fetch_yf_bars()`  
**Problem:** yfinance returns the current in-progress 1-min bar with `volume = 0` (bar hasn't closed yet). Every strategy computed `vol_r = 0 / vol_ma = 0.0x`. Strategies needing vol > 1.3–1.5x were permanently blocked:
- Momentum Flip (needed 1.5x) — dead
- VWAP Reclaim (needed 1.3x) — dead
- Trend Breakout (needed 1.4x) — dead
- ORB Breakout (needed 1.5x) — dead

**Fix:**
```python
# Drop last bar if volume == 0 (in-progress bar)
if len(df) > 1 and df["volume"].iloc[-1] == 0:
    df = df.iloc[:-1].reset_index(drop=True)
```

---

### Bug 3 — Momentum Flip required same-bar EMA + MACD cross (structurally unfireable)
**File:** `trade_strategies.py` → `strategy_momentum_flip()`  
**Problem:** Required EMA9/21 AND MACD(12,26,9) to cross on the **exact same 1-minute bar**. MACD always lags EMA by 5–15 bars on 1-min data. In 347 scans today, this never aligned once.  
**Fix:** Changed to 3-bar lookback window — either cross triggers, both must currently agree on direction. Vol threshold lowered 2.0x → 1.5x.

```python
# Either cross happened in last 3 bars + both currently aligned
(ema_xup_recent or macd_xup_recent) and ema_bull_now and macd_bull_now and vol >= 1.5x
```

---

### Bug 4 — ORB only checked last 2 bars (missed breaks that happened before bot started)
**File:** `trade_strategies.py` → `strategy_orb()`  
**Problem:** Bot started at 11:11 AM; ORB break happened at ~10:05 AM. Condition `p["close"] <= orb_high` was permanently False for all subsequent bars.  
**Fix:** Look back 10 bars for the break, gated by current price still being on the break side.

---

### Bug 5 — BB+ADX Reversal fired SHORT into an uptrend
**File:** `trade_strategies.py` → `strategy_bb_adx_reversal()`  
**Problem:** At 2:50 PM, ADX was 23 (appeared ranging) but market was in a strong post-tweet uptrend. Strategy fired a SHORT at $736.15 while SPY continued to $739+. ADX lags and reads "ranging" during consolidation after a strong move.  
**Fixes:**
1. Added EMA regime filter: no SHORT when `ema9 > ema21`, no LONG when `ema9 < ema21`
2. Relaxed ADX threshold 25 → 30

---

### Bug 6 — Exit Manager hallucinated +51% profit on a losing trade
**File:** `exit_manager.py` → `evaluate()`  
**Problem:** Entry premium was calculated as `ATR × 0.5 = $0.19`. At exit, ATR had grown from 0.39 → 0.57 (market moving more = larger ATR). Bot computed `$0.285 / $0.19 = 1.5x = "+51%"`. This has zero relation to actual option P&L. The $735 PUT with SPY at $739 was nearly worthless.  
**Fix:** Removed all ATR-based premium tracking. Exit timing driven by price action only. Display now shows real underlying move: `$3.26 against you`.

---

## What's Still Not Right (Next Session)

### 1. Volume thresholds too high for SPY
Only 6 scans crossed 1.3x vol all day. SPY is so liquid that 1.3–1.5x on a 1-min bar is rare. Need SPY-specific thresholds (~0.8x). **This is what the backtest is for.**

### 2. Polling doesn't align to bar closes
Bot polls every 15 seconds. EMA/MACD crosses happen mid-bar and are gone before next poll hits. Fix: poll every 60 seconds aligned to the minute close.

---

## The Backtest Task (Pending)

### What to run locally:
```bash
cd "C:\Users\Sam Bertolina\OneDrive\Desktop\quantdesk-bot"
git pull origin main
python backtest_agent.py --symbol SPY --days 500 --min-freq 1.0
```

### What it does:
- Fetches 500 days of SPY 1-min bars from yfinance
- Sweeps vol threshold 0.0x → 2.0x in 0.1x steps for every strategy
- Ranks by **consistency score = signals_per_day × win_rate** (not just edge)
- Walk-forward validates on held-out last 20% of data
- Minimum gate: must fire ≥ 1.0x per day to qualify

### Output files:
- `backtest_results.csv` — full sweep, all thresholds tested
- `backtest_recommended.csv` — **upload this back to Claude** → auto-updates strategy parameters

### What the recommended CSV contains:
| Column | Description |
|---|---|
| `strategy` | Strategy name (matches code) |
| `side` | long / short |
| `recommended_vol_threshold` | Vol threshold to use in code |
| `test_signals_per_day` | Avg signals/day on held-out data |
| `test_win_rate` | Win rate on held-out data |
| `test_edge` | Avg ATR-normalized return |
| `test_consistency` | signals_per_day × win_rate |
| `validated` | True/False — only use True rows |
| `note` | Human-readable recommendation |

When you upload `backtest_recommended.csv`, Claude will read it and update the vol thresholds in `trade_strategies.py` directly.

---

## Strategy Status Summary

| Strategy | Status | Notes |
|---|---|---|
| Momentum Flip | ✅ Fixed | 3-bar window, vol 1.5x |
| VWAP Reclaim | ✅ Unblocked | vol bug fixed |
| Trend Breakout | ✅ Unblocked | vol bug fixed |
| ORB Breakout | ✅ Fixed | 10-bar lookback |
| Power Hour Dip | ✅ Working | 3–3:55 PM only, no vol gate |
| Keltner Bounce | ✅ Fixed | KC columns now computed |
| BB+ADX Reversal | ✅ Fixed | EMA regime filter added |

---

## Files Changed This Session

| File | What changed |
|---|---|
| `trading_alert_agent.py` | KC indicators added; zero-vol bar dropped |
| `trade_strategies.py` | Momentum Flip 3-bar window; ORB 10-bar lookback; BB+ADX EMA filter + ADX 25→30 |
| `exit_manager.py` | Fake ATR P&L removed; real underlying move shown instead |
| `backtest_agent.py` | Full rewrite — consistency-first, vol sweep, session VWAP |

All changes pushed to `main` on `sambone213-cloud/trading-signal-telegram-bot`.

---

## Key Numbers From Today's Logs

- Bot ran: 13:04 – 15:59 ET
- SPY range: $725.39 – $739.91
- Total scans: 691
- Signals fired: **1** (BB+ADX Reversal SHORT at $736.15 — the only strategy with no vol gate)
- Vol >= 1.3x: only 6 scans all day
- ADX at signal time: 23 | ADX at exit time: 77 (market was trending hard, not ranging)
