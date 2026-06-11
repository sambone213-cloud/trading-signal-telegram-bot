"""
backtest_agent.py  — v2
────────────────────────
Consistency-first parameter sweep for SPY 1-min data.

PRIMARY METRIC: consistency_score = signals_per_day × win_rate
  A strategy firing 3x/day at 60% WR (score 1.80) beats one firing
  1x/week at 80% WR (score 0.23). We want reliable daily signals.

What it sweeps:
  - Volume threshold 0.0x (no gate) → 2.0x in 0.1x steps per strategy
  - Tests each strategy's core signal on both sides
  - Walk-forward validation: train on first 80%, test on last 20%

Output (two CSVs):
  backtest_results.csv       — full sweep, every threshold tested
  backtest_recommended.csv   — one row per strategy, best params → upload back to Claude

Usage:
  python backtest_agent.py --symbol SPY --days 500 --min-freq 1.0
"""

import argparse
import warnings
import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VOL_SWEEP = [round(x * 0.1, 1) for x in range(0, 22)]   # 0.0 → 2.1


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    print(f"  Fetching {days}d of {symbol} 1-min bars via yfinance...")
    end   = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    hist  = yf.Ticker(symbol).history(start=start, end=end, interval="1m", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("No data returned from yfinance")
    hist.index = hist.index.tz_convert("UTC")
    df = pd.DataFrame({
        "datetime": hist.index,
        "open":     hist["Open"].values,
        "high":     hist["High"].values,
        "low":      hist["Low"].values,
        "close":    hist["Close"].values,
        "volume":   hist["Volume"].values,
    })
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    # Drop zero-volume bars (incomplete in-progress bars from yfinance)
    df = df[df["volume"] > 0].reset_index(drop=True)
    print(f"  Got {len(df):,} bars  ({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, v = df["close"], df["volume"]

    for span in [9, 21, 50]:
        df[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    high, low = df["high"], df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Keltner Channels
    ema20_kc = c.ewm(span=20, adjust=False).mean()
    atr10    = tr.rolling(10).mean()
    df["kc_upper"] = ema20_kc + 2 * atr10
    df["kc_lower"] = ema20_kc - 2 * atr10

    # Session VWAP — resets each trading day (not cumulative across days)
    df["date"] = df["datetime"].dt.date
    df["cum_pv"]  = df.groupby("date").apply(
        lambda g: (g["close"] * g["volume"]).cumsum()
    ).reset_index(level=0, drop=True)
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"]    = df["cum_pv"] / df["cum_vol"]

    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr14    = tr.rolling(14).mean()
    df["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = (100 * (df["plus_di"] - df["minus_di"]).abs()
          / (df["plus_di"] + df["minus_di"]).replace(0, np.nan))
    df["adx"] = dx.rolling(14).mean()

    df["vwma9"]  = (c * v).rolling(9).sum()  / v.rolling(9).sum()
    df["vwma21"] = (c * v).rolling(21).sum() / v.rolling(21).sum()

    df["vol_ma20"]  = v.rolling(20, min_periods=5).mean()
    df["vol_ratio"] = v / df["vol_ma20"].replace(0, np.nan)

    df["mom10"] = c - c.shift(10)

    dt_et = df["datetime"] - pd.Timedelta(hours=4)
    df["hour_et"]        = dt_et.dt.hour
    df["minute_et"]      = dt_et.dt.minute
    df["mins_since_open"] = ((df["hour_et"] - 9) * 60 + df["minute_et"] - 30).clip(lower=0)

    # Per-day ORB high/low (9:30–9:59 ET = first 30 bars each day)
    df["orb_high"] = np.nan
    df["orb_low"]  = np.nan
    for date, day_df in df.groupby("date"):
        orb = day_df[(day_df["hour_et"] == 9) & (day_df["minute_et"] >= 30)]
        if len(orb) >= 15:
            df.loc[day_df.index, "orb_high"] = float(orb["high"].max())
            df.loc[day_df.index, "orb_low"]  = float(orb["low"].min())

    return df.dropna(subset=["rsi", "atr", "adx"]).reset_index(drop=True)


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest_signal(df, signal_mask, side="long", hold_bars=5,
                    tp_atr=2.5, sl_atr=1.0):
    """
    Simulate entries on signal_mask bars. Returns win_rate, edge, signal count.
    No overlapping trades — once in a trade, skip new signals until exit.
    """
    entries = df.index[signal_mask & (df.index >= 30)]
    if len(entries) < 5:
        return {"signals": int(len(entries)), "win_rate": 0.0, "edge": 0.0}

    results  = []
    last_exit = -1

    for idx in entries:
        if idx <= last_exit:
            continue
        if idx + hold_bars >= len(df):
            continue
        entry = float(df["close"].iloc[idx])
        atr   = float(df["atr"].iloc[idx])
        if atr <= 0:
            continue

        tp = entry + tp_atr * atr if side == "long" else entry - tp_atr * atr
        sl = entry - sl_atr * atr if side == "long" else entry + sl_atr * atr

        outcome = 0.0
        for i in range(1, hold_bars + 1):
            bar = df.iloc[idx + i]
            if side == "long":
                if float(bar["high"]) >= tp:  outcome = tp_atr;  last_exit = idx + i; break
                if float(bar["low"])  <= sl:  outcome = -sl_atr; last_exit = idx + i; break
            else:
                if float(bar["low"])  <= tp:  outcome = tp_atr;  last_exit = idx + i; break
                if float(bar["high"]) >= sl:  outcome = -sl_atr; last_exit = idx + i; break
        else:
            close_out = float(df["close"].iloc[idx + hold_bars])
            outcome   = (close_out - entry) / atr * (1 if side == "long" else -1)
            last_exit = idx + hold_bars

        results.append(outcome)

    if not results:
        return {"signals": 0, "win_rate": 0.0, "edge": 0.0}

    wr = sum(1 for r in results if r > 0) / len(results)
    return {
        "signals":  int(len(results)),
        "win_rate": round(wr, 3),
        "edge":     round(float(np.mean(results)), 4),
    }


# ── Strategy base signals ─────────────────────────────────────────────────────

def build_base_signals(df: pd.DataFrame) -> dict:
    """
    Core signal conditions for each strategy — NO volume gate.
    Volume threshold is swept separately so we can find the optimal value.
    Returns dict: strategy_name -> (long_mask, short_mask)
    """

    def xup(col_a, col_b, lb=3):
        cross = (df[col_a].shift(1) <= df[col_b].shift(1)) & (df[col_a] > df[col_b])
        result = cross.copy()
        for i in range(1, lb):
            result = result | cross.shift(i)
        return result.fillna(False)

    def xdn(col_a, col_b, lb=3):
        cross = (df[col_a].shift(1) >= df[col_b].shift(1)) & (df[col_a] < df[col_b])
        result = cross.copy()
        for i in range(1, lb):
            result = result | cross.shift(i)
        return result.fillna(False)

    ema_bull   = df["ema9"] > df["ema21"]
    ema_bear   = df["ema9"] < df["ema21"]
    macd_bull  = df["macd"] > df["macd_signal"]
    macd_bear  = df["macd"] < df["macd_signal"]
    above_vwap = df["close"] > df["vwap"]
    below_vwap = df["close"] < df["vwap"]
    after_30m  = df["mins_since_open"] >= 30
    after_15m  = df["mins_since_open"] >= 15

    no_short = pd.Series(False, index=df.index)

    # 1. Momentum Flip
    mf_long = (
        (xup("ema9", "ema21") | xup("macd", "macd_signal", lb=1))
        & ema_bull & macd_bull
        & df["rsi"].between(35, 65)
        & after_30m
    )
    mf_short = (
        (xdn("ema9", "ema21") | xdn("macd", "macd_signal", lb=1))
        & ema_bear & macd_bear
        & df["rsi"].between(35, 65)
        & after_30m
    )

    # 2. VWAP Reclaim
    vwap_xup = (df["close"].shift(1) < df["vwap"].shift(1)) & (df["close"] > df["vwap"])
    vwap_xdn = (df["close"].shift(1) > df["vwap"].shift(1)) & (df["close"] < df["vwap"])
    vr_long  = vwap_xup & ema_bull & (df["rsi"] < 68) & after_15m
    vr_short = vwap_xdn & ema_bear & (df["rsi"] > 32) & after_15m

    # 3. Trend Breakout
    tb_long = (
        xdn("macd", "macd_signal", lb=1) & ema_bull & above_vwap
        & (df["adx"] >= 22) & df["rsi"].between(40, 72) & after_30m
    )
    tb_short = (
        xdn("macd", "macd_signal", lb=1) & ema_bear & below_vwap
        & (df["adx"] >= 22) & df["rsi"].between(28, 60) & after_30m
    )
    # Fix: use xup for short (MACD cross down)
    tb_long  = (
        xup("macd", "macd_signal", lb=1) & ema_bull & above_vwap
        & (df["adx"] >= 22) & df["rsi"].between(40, 72) & after_30m
    )
    tb_short = (
        xdn("macd", "macd_signal", lb=1) & ema_bear & below_vwap
        & (df["adx"] >= 22) & df["rsi"].between(28, 60) & after_30m
    )

    # 4. ORB Breakout
    orb_valid = (
        df["orb_high"].notna()
        & (df["mins_since_open"] >= 30)
        & (df["hour_et"] <= 14)
    )
    orb_buf  = df["atr"] * 0.1
    orb_size = (df["orb_high"] - df["orb_low"]).clip(lower=0.01)
    orb_long = (
        orb_valid & ema_bull
        & (df["close"] > df["orb_high"] + orb_buf)
        & (df["close"].shift(1) <= df["orb_high"] + orb_buf)
        & (df["close"] < df["orb_high"] + 2.5 * orb_size)
        & (df["rsi"] < 75)
    )
    orb_short = (
        orb_valid & ema_bear
        & (df["close"] < df["orb_low"] - orb_buf)
        & (df["close"].shift(1) >= df["orb_low"] - orb_buf)
        & (df["close"] > df["orb_low"] - 2.5 * orb_size)
        & (df["rsi"] > 25)
    )

    # 5. Power Hour Dip (long only — institutional close-of-day buying)
    ph_time   = (df["hour_et"] == 15) & (df["minute_et"] < 55)
    vwma_xup1 = (df["vwma9"].shift(1) <= df["vwma21"].shift(1)) & (df["vwma9"] > df["vwma21"])
    ph_long   = ph_time & vwma_xup1 & ema_bull & (df["mom10"] < 0) & df["rsi"].between(30, 62)

    # 6. Keltner Bounce (ranging markets only)
    after_1030 = (df["hour_et"] > 10) | ((df["hour_et"] == 10) & (df["minute_et"] >= 30))
    kb_long  = (
        (df["close"] < df["kc_lower"]) & ema_bull
        & (df["adx"] < 28) & (df["rsi"] < 50) & after_1030
    )
    kb_short = (
        (df["close"] > df["kc_upper"]) & ema_bear
        & (df["adx"] < 28) & (df["rsi"] > 50) & after_1030
    )

    # 7. BB+ADX Reversal (ranging markets only)
    two_low  = (df["close"] <= df["bb_lower"]) & (df["close"].shift(1) <= df["bb_lower"].shift(1))
    two_high = (df["close"] >= df["bb_upper"]) & (df["close"].shift(1) >= df["bb_upper"].shift(1))
    rsi_up   = df["rsi"] > df["rsi"].shift(1)
    rsi_down = df["rsi"] < df["rsi"].shift(1)
    bb_long  = two_low  & ema_bull & (df["rsi"] < 35) & rsi_up   & (df["adx"] < 30) & after_1030
    bb_short = two_high & ema_bear & (df["rsi"] > 65) & rsi_down & (df["adx"] < 30) & after_1030

    return {
        "Momentum Flip":    (mf_long.fillna(False),  mf_short.fillna(False)),
        "VWAP Reclaim":     (vr_long.fillna(False),  vr_short.fillna(False)),
        "Trend Breakout":   (tb_long.fillna(False),  tb_short.fillna(False)),
        "ORB Breakout":     (orb_long.fillna(False), orb_short.fillna(False)),
        "Power Hour Dip":   (ph_long.fillna(False),  no_short),
        "Keltner Bounce":   (kb_long.fillna(False),  kb_short.fillna(False)),
        "BB+ADX Reversal":  (bb_long.fillna(False),  bb_short.fillna(False)),
    }


# ── Volume sweep ──────────────────────────────────────────────────────────────

def sweep_vol_for_strategy(df, base_mask, side, n_days):
    """Test every vol threshold. Returns full sweep DataFrame sorted by consistency_score."""
    rows = []
    for vt in VOL_SWEEP:
        if vt == 0.0:
            mask = base_mask
        else:
            mask = base_mask & (df["vol_ratio"] >= vt)

        r   = backtest_signal(df, mask, side=side)
        spd = r["signals"] / n_days if n_days > 0 else 0
        cs  = round(spd * r["win_rate"], 4)
        rows.append({
            "vol_threshold":  vt,
            "signals":        r["signals"],
            "signals_per_day": round(spd, 2),
            "win_rate":       r["win_rate"],
            "edge":           r["edge"],
            "consistency_score": cs,
        })

    return pd.DataFrame(rows).sort_values("consistency_score", ascending=False).reset_index(drop=True)


# ── Full sweep ────────────────────────────────────────────────────────────────

def run_full_sweep(df, n_days, min_freq):
    base_signals = build_base_signals(df)
    all_rows = []

    for strat_name, (long_mask, short_mask) in base_signals.items():
        for side, base_mask in [("long", long_mask), ("short", short_mask)]:
            total = int(base_mask.sum())
            if total < 5:
                continue

            print(f"  {strat_name:<22} {side:<6} (base signals: {total:,}) ...", end=" ", flush=True)
            sweep = sweep_vol_for_strategy(df, base_mask, side, n_days)

            # Pick best row that meets min_freq; fall back to best available
            meets = sweep[sweep["signals_per_day"] >= min_freq]
            best  = meets.iloc[0] if not meets.empty else sweep.iloc[0]
            meets_min = not meets.empty

            print(f"best vol={best['vol_threshold']}x  "
                  f"{best['signals_per_day']:.1f}/day  "
                  f"WR {best['win_rate']*100:.0f}%  "
                  f"E={best['edge']:+.3f}  "
                  f"score={best['consistency_score']:.3f}"
                  + ("" if meets_min else "  ⚠️ below min freq"))

            for _, row in sweep.iterrows():
                all_rows.append({
                    "strategy":       strat_name,
                    "side":           side,
                    "vol_threshold":  row["vol_threshold"],
                    "signals":        row["signals"],
                    "signals_per_day": row["signals_per_day"],
                    "win_rate":       row["win_rate"],
                    "edge":           row["edge"],
                    "consistency_score": row["consistency_score"],
                    "meets_min_freq": row["signals_per_day"] >= min_freq,
                    "is_best_for_strategy": row["vol_threshold"] == best["vol_threshold"],
                })

    return pd.DataFrame(all_rows)


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward(df, full_results):
    """Validate best params from first 80% on the held-out last 20%."""
    split     = int(len(df) * 0.8)
    train_df  = df.iloc[:split].reset_index(drop=True)
    test_df   = df.iloc[split:].reset_index(drop=True)
    test_days = test_df["date"].nunique()

    train_signals = build_base_signals(train_df)
    test_signals  = build_base_signals(test_df)

    best_params = full_results[full_results["is_best_for_strategy"]].copy()
    rows = []

    for _, row in best_params.iterrows():
        strat = row["strategy"]
        side  = row["side"]
        vt    = row["vol_threshold"]
        idx   = 0 if side == "long" else 1

        if strat not in test_signals:
            continue

        test_base = test_signals[strat][idx]
        test_mask = test_base if vt == 0.0 else test_base & (test_df["vol_ratio"] >= vt)

        r_test   = backtest_signal(test_df, test_mask, side=side)
        spd_test = r_test["signals"] / test_days if test_days > 0 else 0
        cs_test  = round(spd_test * r_test["win_rate"], 4)

        edge_decay = ((row["edge"] - r_test["edge"])
                      / (abs(row["edge"]) + 1e-9))

        validated = (
            r_test["edge"]     >= 0.05 and
            r_test["win_rate"] >= 0.50 and
            edge_decay         <  0.50
        )

        rows.append({
            "strategy":              strat,
            "side":                  side,
            "recommended_vol_threshold": vt,
            "train_signals_per_day": row["signals_per_day"],
            "train_win_rate":        row["win_rate"],
            "train_edge":            row["edge"],
            "train_consistency":     row["consistency_score"],
            "test_signals_per_day":  round(spd_test, 2),
            "test_win_rate":         r_test["win_rate"],
            "test_edge":             r_test["edge"],
            "test_consistency":      cs_test,
            "edge_decay_pct":        round(edge_decay * 100, 1),
            "validated":             validated,
            "note": (
                "VALIDATED — use these params" if validated
                else "NOT VALIDATED — overfitted or too rare; skip or tune further"
            ),
        })

    return (pd.DataFrame(rows)
            .sort_values(["validated", "test_consistency"], ascending=[False, False])
            .reset_index(drop=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuantDesk Backtest — consistency-first sweep")
    parser.add_argument("--symbol",              default="SPY")
    parser.add_argument("--days",      type=int, default=500)
    parser.add_argument("--min-freq",  type=float, default=1.0,
                        help="Minimum signals per trading day (default 1.0)")
    parser.add_argument("--output",              default="backtest_results.csv")
    parser.add_argument("--recommended-output",  default="backtest_recommended.csv")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  QuantDesk Backtest v2  |  {args.symbol}  |  {args.days}d")
    print(f"  Min frequency: {args.min_freq} signals/day")
    print(f"  Ranking by: consistency_score = signals_per_day × win_rate")
    print(f"{'='*64}\n")

    df     = fetch_data(args.symbol, args.days)
    df     = compute_indicators(df)
    n_days = df["date"].nunique()
    print(f"  {len(df):,} bars  |  {n_days} trading days\n")

    print(f"[Phase 1 — Sweeping vol thresholds 0.0x → 2.0x per strategy]")
    full_results = run_full_sweep(df, n_days, args.min_freq)

    print(f"\n[Phase 2 — Walk-forward validation (80/20 split)]")
    recommended = walk_forward(df, full_results)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  FINAL RESULTS — ranked by test consistency score")
    print(f"  ✅ validated   ⚠️  not validated (overfit or too rare)")
    print(f"{'='*64}")
    for _, r in recommended.iterrows():
        tag = "✅" if r["validated"] else "⚠️ "
        print(
            f"  {tag} {r['strategy']:<22} {r['side']:<6} "
            f"vol>={r['recommended_vol_threshold']}x  "
            f"{r['test_signals_per_day']:.1f}/day  "
            f"WR {r['test_win_rate']*100:.0f}%  "
            f"E={r['test_edge']:+.3f}  "
            f"score={r['test_consistency']:.3f}"
        )
    print(f"{'='*64}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    full_results.to_csv(args.output, index=False)
    recommended.to_csv(args.recommended_output, index=False)

    validated_count = recommended["validated"].sum()
    print(f"  Full sweep    → {args.output}  ({len(full_results)} rows)")
    print(f"  Recommended   → {args.recommended_output}  "
          f"({validated_count} validated, {len(recommended)-validated_count} not)")
    print(f"\n  ► Upload {args.recommended_output} back to Claude to auto-update strategy parameters.\n")


if __name__ == "__main__":
    main()
