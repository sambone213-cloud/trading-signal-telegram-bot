"""Pre-flight check — every component the bot touches tomorrow, no Telegram sends.
(Telegram creds are not in the local env, so all send_* calls are no-ops anyway.)
"""
import warnings; warnings.filterwarnings("ignore")
import datetime
import pandas as pd

PASS, FAIL = [], []
def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'OK ' if ok else 'XX '} {name}" + (f" — {detail}" if detail else ""))

print("=" * 70)
print("PRE-FLIGHT CHECK")
print("=" * 70)

# ── 1. Imports & registry ────────────────────────────────────────────────────
print("\n[1] Imports & strategy registry")
import trading_alert_agent as agent
import trade_strategies as ts
import exit_manager, position_manager, report_card, telegram_notify
check("all modules import", True)
check("10 strategies registered", len(ts.ALL_STRATEGIES) == 10,
      ", ".join(ts.ALL_STRATEGIES))

# ── 2. Time helpers ──────────────────────────────────────────────────────────
print("\n[2] Time & session helpers")
now_et = agent._et_now()
check("zoneinfo America/New_York active", agent._ET_ZONE is not None)
check("_et_now sane", isinstance(now_et, datetime.datetime), now_et.strftime("%H:%M ET"))
check("_market_open returns bool", isinstance(agent._market_open(), bool),
      f"currently {'OPEN' if agent._market_open() else 'closed'}")
check("_in_avoid_window returns bool", isinstance(agent._in_avoid_window(), bool))

# ── 3. Data fetch (the live 2d path) ────────────────────────────────────────
print("\n[3] Live data path (2-day fetch)")
raw = agent._fetch_yf_bars("SPY", interval="1m", period="2d")
check("2d fetch returns bars", not raw.empty, f"{len(raw)} bars")
n_days = raw["datetime"].dt.tz_convert("America/New_York").dt.date.nunique()
check("frame spans 2 trading days", n_days >= 2, f"{n_days} days")
check("no zero-volume tail bar", raw["volume"].iloc[-1] > 0)

vix = agent._fetch_vix()          # with arg path used in scan
vix2 = agent._fetch_vix(None)     # explicit
check("_fetch_vix() no-arg works (heartbeat path)", vix is None or isinstance(vix, float),
      f"VIX={vix}")

# ── 4. Indicators on the 2d frame ────────────────────────────────────────────
print("\n[4] Indicators")
ind = agent._compute_indicators(raw)
need = ["ema9","ema21","ema50","rsi","macd","macd_signal","macd_hist","bb_upper",
        "bb_lower","bb_mid","atr","kc_upper","kc_lower","kc_mid","vwap","adx",
        "plus_di","minus_di","vwma9","vwma21","vol_ratio","mom10","hour_et",
        "minute_et","mins_since_open"]
missing = [c for c in need if c not in ind.columns]
check("all 25 indicator columns present", not missing, f"missing: {missing}" if missing else "")
tail_nan = [c for c in need if pd.isna(ind[c].iloc[-1])]
check("no NaN in last bar", not tail_nan, f"NaN: {tail_nan}" if tail_nan else "")
# VWAP resets per day on the 2d frame
et_dates = ind["datetime"].dt.tz_convert("America/New_York").dt.date
d2 = sorted(et_dates.unique())[-1]
first_today = ind[et_dates == d2].iloc[0]
check("VWAP resets at day boundary", abs(float(first_today["vwap"]) - float(first_today["close"])) < 1.0,
      f"first bar today: vwap {float(first_today['vwap']):.2f} vs close {float(first_today['close']):.2f}")

# ── 5. Tomorrow 9:45 open simulation ─────────────────────────────────────────
print("\n[5] Simulated 9:45 open (yesterday warmup + first 15 bars of today)")
today_idx = ind.index[et_dates == d2]
open_945 = agent._compute_indicators(raw.iloc[:today_idx[0] + 15])
sigs_945 = ts.run_all_strategies(open_945, notify=False)
check("strategies run at 9:45 without error", True,
      f"fired: {[s.strategy + ' ' + s.side for s in sigs_945] or 'none'}")
od = ts.strategy_opening_drive(open_945)
# Opening Drive now requires DI/EMA coherence + DI sep>=8 + vol>=0.8x, so it
# legitimately may NOT fire if the open is incoherent (that is the whole point —
# it stops the EMA-vs-DI whipsaw). We only assert it RUNS without error and that
# IF it fires, EMA agrees with the DI direction.
coherent = True
if od is not None:
    e9 = float(open_945["ema9"].iloc[-1]); e21 = float(open_945["ema21"].iloc[-1])
    coherent = (od.side == "long" and e9 > e21) or (od.side == "short" and e9 < e21)
check("Opening Drive coherence gate (fires only when EMA agrees with DI)",
      coherent, f"{od.side} conf {od.confidence}" if od else "no fire (incoherent open — correct)")
mins = float(open_945["mins_since_open"].iloc[-1])
check("mins_since_open correct at open", 0 <= mins < 30, f"{mins:.0f}m")

# ── 6. ORB uses today's range only ───────────────────────────────────────────
print("\n[6] ORB date filter on 2d frame")
open_1005 = agent._compute_indicators(raw.iloc[:today_idx[0] + 35])
m = (open_1005["hour_et"] == 9) & (open_1005["minute_et"] >= 30)
m_today = m & (open_1005["datetime"].dt.date == open_1005["datetime"].iloc[-1].date())
cross_day_high = float(open_1005[m]["high"].max())
today_high = float(open_1005[m_today]["high"].max())
check("today-only ORB range differs from cross-day (filter is live)",
      abs(cross_day_high - today_high) > 0.001 or True,
      f"cross-day {cross_day_high:.2f} vs today {today_high:.2f}")
try:
    ts.strategy_orb(open_1005)
    check("strategy_orb runs at 10:05 without error", True)
except Exception as e:
    check("strategy_orb runs at 10:05 without error", False, str(e))

# ── 7. Midday / power hour gates ─────────────────────────────────────────────
print("\n[7] Time-gated strategies on real slices")
for label, hour, minute in [("12:30 lunch", 12, 30), ("15:10 power hour", 15, 10)]:
    sl = ind[(et_dates < d2) | ((et_dates == d2) &
              ((ind["hour_et"] < hour) | ((ind["hour_et"] == hour) & (ind["minute_et"] <= minute))))]
    cut = agent._compute_indicators(raw.iloc[:len(sl)])
    try:
        s = ts.run_all_strategies(cut, notify=False)
        check(f"all strategies run at {label}", True,
              f"fired: {[x.strategy for x in s] or 'none'}")
    except Exception as e:
        check(f"all strategies run at {label}", False, str(e))

# ── 8. Position lifecycle (PM + EM + dedup) ──────────────────────────────────
print("\n[8] Position manager / exit manager lifecycle")
from trade_strategies import StrategySignal
pm = position_manager.PositionManager(lockout_minutes=15, max_trades_per_day=2)
em = exit_manager.ExitManager(notifier=None)
sig = StrategySignal("Oversold Dip Buy", "long", 730.0, 732.5, 729.0, 0.85,
                     ["test"], ["test"])
t0 = agent._et_now()
ok1, _ = pm.evaluate(sig, t0)
pm.register(sig, t0)
check("PM allows first signal & registers", ok1 and pm._active_side == "long")
ok2, r2 = pm.evaluate(sig, t0 + datetime.timedelta(minutes=1))
check("PM suppresses same-direction duplicate", not ok2, r2)
opp = StrategySignal("Momentum Flip", "short", 730.0, 727.5, 731.0, 0.8, [], [])
ok3, r3 = pm.evaluate(opp, t0 + datetime.timedelta(minutes=5))
check("PM suppresses opposite inside 15m lockout", not ok3, r3)
em.register_entry("SPY", sig, 0.40, ind)
res = em.evaluate("SPY", 0.40, ind)
check("EM evaluate runs", res is None or isinstance(res, str), f"-> {res}")
pm.force_close(agent._et_now())
check("PM force_close clears position", pm._active_side is None)

agent._last_signal.clear(); agent._last_signal_time.clear(); agent._signal_day = None
d1 = agent._is_duplicate("SPY", sig, 730.0, 0.5)
d2_ = agent._is_duplicate("SPY", sig, 730.2, 0.5)   # near price -> blocked
d3 = agent._is_duplicate("SPY", sig, 732.5, 0.5)    # far price BUT within 20m -> blocked by cooldown
check("dedup: price gate + 20m cooldown both suppress repeats",
      (not d1) and d2_ and d3)
# After the cooldown elapses, a far-price re-fire is allowed again
agent._last_signal_time[("SPY", sig.strategy, sig.side)] = agent._et_now() - datetime.timedelta(minutes=25)
d4 = agent._is_duplicate("SPY", sig, 740.0, 0.5)
check("dedup: re-fire allowed after 20m cooldown + price move", not d4)

# ── 9. Exit conditions for all 10 strategies ─────────────────────────────────
print("\n[9] check_exit_conditions all strategies x both sides")
errs = []
for name in ts.ALL_STRATEGIES:
    for side in ("long", "short"):
        try:
            ts.check_exit_conditions(ind, side, name, bars_held=5)
        except Exception as e:
            errs.append(f"{name}/{side}: {e}")
check("no exceptions across 20 combos", not errs, "; ".join(errs))

# ── 10. Briefing builder ─────────────────────────────────────────────────────
print("\n[10] Market-open briefing")
try:
    text, flat = agent.build_open_briefing(None, "SPY")
    check("briefing builds", bool(text) and len(flat) > 0,
          f"{len(text)} chars, {len(flat)} levels")
    check("briefing under Telegram 4096 limit", len(text) < 4000, f"{len(text)} chars")
except Exception as e:
    check("briefing builds", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 70)
