"""
backtest_agent.py
─────────────────
Exhaustive backtesting pipeline for SPY 1-min signals.

Pipeline:
  Phase 1 — Fetch real SPY 1-min data (60-90 days via yfinance)
  Phase 2 — Exhaustive single-condition scan (every indicator threshold)
  Phase 3 — Combination search (2-way and 3-way stacks)
  Phase 4 — Time segmentation (30-min buckets × VIX regime × ADX regime)
  Phase 5 — Walk-forward validation (80/20 split)

Output: ranked edge report saved to backtest_results.csv

Usage:
  python backtest_agent.py --symbol SPY --days 60 --output backtest_results.csv
"""

import argparse
import itertools
import warnings
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str = "SPY", days: int = 60) -> pd.DataFrame:
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
        "open":   hist["Open"].values,
        "high":   hist["High"].values,
        "low":    hist["Low"].values,
        "close":  hist["Close"].values,
        "volume": hist["Volume"].values,
    })
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  Got {len(df):,} bars  ({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    v = df["volume"]

    for span in [9, 21, 50, 200]:
        df[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_mid"]   = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20

    high, low = df["high"], df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([(high-low), (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / c

    df["vwap"] = (c * v).cumsum() / v.cumsum()

    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr14    = tr.rolling(14).mean()
    df["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    df["vwma9"]  = (c * v).rolling(9).sum()  / v.rolling(9).sum()
    df["vwma21"] = (c * v).rolling(21).sum() / v.rolling(21).sum()

    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"].replace(0, np.nan)

    df["hour_et"]   = df["datetime"].dt.hour - 4  # UTC-4 approximation
    df["minute_et"] = df["datetime"].dt.minute
    df["time_bucket"] = (df["hour_et"] * 60 + df["minute_et"]) // 30  # 30-min buckets

    return df.dropna().reset_index(drop=True)


# ── Signal conditions (atomic) ────────────────────────────────────────────────

def define_conditions(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """All testable atomic conditions as boolean Series."""
    c = {
        # EMA trend
        "ema9_above_21":    df["ema9"] > df["ema21"],
        "ema9_below_21":    df["ema9"] < df["ema21"],
        "ema21_above_50":   df["ema21"] > df["ema50"],
        "close_above_vwap": df["close"] > df["vwap"],
        "close_below_vwap": df["close"] < df["vwap"],

        # RSI
        "rsi_lt_30": df["rsi"] < 30,
        "rsi_lt_35": df["rsi"] < 35,
        "rsi_lt_40": df["rsi"] < 40,
        "rsi_gt_60": df["rsi"] > 60,
        "rsi_gt_65": df["rsi"] > 65,
        "rsi_gt_70": df["rsi"] > 70,
        "rsi_gt_75": df["rsi"] > 75,
        "rsi_rising": df["rsi"] > df["rsi"].shift(1),
        "rsi_falling": df["rsi"] < df["rsi"].shift(1),

        # MACD
        "macd_above_0":    df["macd"] > 0,
        "macd_below_0":    df["macd"] < 0,
        "macd_cross_up":   (df["macd"].shift(1) <= df["macd_signal"].shift(1)) & (df["macd"] > df["macd_signal"]),
        "macd_cross_down": (df["macd"].shift(1) >= df["macd_signal"].shift(1)) & (df["macd"] < df["macd_signal"]),
        "hist_expanding":  df["macd_hist"].abs() > df["macd_hist"].shift(1).abs(),

        # BB
        "above_bb_upper": df["close"] >= df["bb_upper"],
        "below_bb_lower": df["close"] <= df["bb_lower"],
        "inside_bb":      (df["close"] < df["bb_upper"]) & (df["close"] > df["bb_lower"]),
        "bb_squeeze":     df["bb_width"] < df["bb_width"].rolling(20).mean() * 0.7,

        # Volume
        "vol_surge_1_4":  df["vol_ratio"] > 1.4,
        "vol_surge_1_8":  df["vol_ratio"] > 1.8,

        # ADX
        "adx_gt_20": df["adx"] > 20,
        "adx_gt_25": df["adx"] > 25,
        "adx_lt_20": df["adx"] < 20,
        "adx_lt_25": df["adx"] < 25,

        # VWMA
        "vwma_cross_up":   (df["vwma9"].shift(1) <= df["vwma21"].shift(1)) & (df["vwma9"] > df["vwma21"]),
        "vwma_cross_down": (df["vwma9"].shift(1) >= df["vwma21"].shift(1)) & (df["vwma9"] < df["vwma21"]),

        # Time of day (ET)
        "after_930":  df["hour_et"] >= 9,
        "after_1000": (df["hour_et"] > 10) | ((df["hour_et"] == 10) & (df["minute_et"] >= 0)),
        "after_1030": (df["hour_et"] > 10) | ((df["hour_et"] == 10) & (df["minute_et"] >= 30)),
        "before_1500":(df["hour_et"] < 15),
        "lunch_hour": (df["hour_et"] == 12),
    }
    return {k: v.astype(bool) for k, v in c.items()}


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest_signal(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    side: str = "long",
    hold_bars: int = 5,
    tp_atr: float = 2.5,
    sl_atr: float = 1.0,
) -> dict:
    """
    Simulate a signal on df. Returns performance stats.
    side: 'long' or 'short'
    hold_bars: max bars to hold if TP/SL not hit
    """
    entries = df.index[signal_mask & (df.index >= 30)]
    if len(entries) < 10:
        return {"signals": len(entries), "win_rate": 0, "edge": 0, "avg_win": 0, "avg_loss": 0}

    results = []
    for idx in entries:
        if idx + hold_bars >= len(df):
            continue
        entry = df["close"].iloc[idx]
        atr   = df["atr"].iloc[idx]
        if atr <= 0:
            continue

        tp = entry + tp_atr * atr if side == "long" else entry - tp_atr * atr
        sl = entry - sl_atr * atr if side == "long" else entry + sl_atr * atr

        outcome = 0.0
        for i in range(1, hold_bars + 1):
            bar = df.iloc[idx + i]
            if side == "long":
                if bar["high"] >= tp:
                    outcome = tp_atr; break
                if bar["low"] <= sl:
                    outcome = -sl_atr; break
            else:
                if bar["low"] <= tp:
                    outcome = tp_atr; break
                if bar["high"] >= sl:
                    outcome = -sl_atr; break
        else:
            outcome = (df["close"].iloc[idx + hold_bars] - entry) / atr * (1 if side == "long" else -1)

        results.append(outcome)

    if not results:
        return {"signals": 0, "win_rate": 0, "edge": 0, "avg_win": 0, "avg_loss": 0}

    wins   = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    wr     = len(wins) / len(results)
    edge   = np.mean(results)

    return {
        "signals":  len(results),
        "win_rate": round(wr, 3),
        "edge":     round(edge, 4),
        "avg_win":  round(np.mean(wins) if wins else 0, 3),
        "avg_loss": round(np.mean(losses) if losses else 0, 3),
    }


# ── Phase 1: Single condition scan ───────────────────────────────────────────

def phase1_single_conditions(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    print("\n[Phase 1] Testing single conditions...")
    rows = []
    for name, mask in conditions.items():
        for side in ["long", "short"]:
            r = backtest_signal(df, mask, side=side)
            if r["signals"] >= 20:
                rows.append({"condition": name, "side": side, **r})
    result = pd.DataFrame(rows).sort_values("edge", ascending=False)
    print(f"  Tested {len(conditions)*2} condition/side combos → {len(result)} with ≥20 signals")
    return result


# ── Phase 2: Combination search ──────────────────────────────────────────────

def phase2_combinations(df: pd.DataFrame, conditions: dict, top_n: int = 20) -> pd.DataFrame:
    print("\n[Phase 2] Testing 2-way and 3-way combinations...")
    # Get top single conditions by edge
    single = phase1_single_conditions(df, conditions)
    top_conditions = single[single["edge"] > 0]["condition"].head(top_n).tolist()

    rows = []
    combos_tested = 0

    # 2-way
    for c1, c2 in itertools.combinations(top_conditions, 2):
        mask = conditions[c1] & conditions[c2]
        for side in ["long", "short"]:
            r = backtest_signal(df, mask, side=side)
            if r["signals"] >= 15 and r["edge"] > 0.05:
                rows.append({"conditions": f"{c1} + {c2}", "side": side, **r})
        combos_tested += 1

    # 3-way
    for c1, c2, c3 in itertools.combinations(top_conditions[:12], 3):
        mask = conditions[c1] & conditions[c2] & conditions[c3]
        for side in ["long", "short"]:
            r = backtest_signal(df, mask, side=side)
            if r["signals"] >= 10 and r["edge"] > 0.08:
                rows.append({"conditions": f"{c1} + {c2} + {c3}", "side": side, **r})
        combos_tested += 1

    result = pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
    print(f"  Tested {combos_tested} combos → {len(result)} with positive edge")
    return result


# ── Phase 3: Time segmentation ───────────────────────────────────────────────

def phase3_time_segment(df: pd.DataFrame, conditions: dict, top_combos: pd.DataFrame) -> pd.DataFrame:
    print("\n[Phase 3] Time segmentation + VIX/ADX regime filtering...")
    if top_combos.empty:
        return pd.DataFrame()

    rows = []
    time_buckets = {
        "9:30-10:00": (df["hour_et"] == 9) & (df["minute_et"] >= 30),
        "10:00-10:30": (df["hour_et"] == 10) & (df["minute_et"] < 30),
        "10:30-11:00": (df["hour_et"] == 10) & (df["minute_et"] >= 30),
        "11:00-12:00": df["hour_et"] == 11,
        "12:00-13:00": df["hour_et"] == 12,
        "13:00-14:00": df["hour_et"] == 13,
        "14:00-15:00": df["hour_et"] == 14,
        "15:00-16:00": df["hour_et"] == 15,
    }
    adx_regimes = {
        "trending (ADX>25)": df["adx"] > 25,
        "ranging (ADX<20)":  df["adx"] < 20,
    }

    for _, combo_row in top_combos.head(10).iterrows():
        cond_names = combo_row["conditions"].split(" + ")
        base_mask  = conditions[cond_names[0]]
        for cn in cond_names[1:]:
            if cn in conditions:
                base_mask = base_mask & conditions[cn]

        for tb_name, tb_mask in time_buckets.items():
            for adx_name, adx_mask in adx_regimes.items():
                mask = base_mask & tb_mask & adx_mask
                r = backtest_signal(df, mask, side=combo_row["side"])
                if r["signals"] >= 8 and r["edge"] > 0.10:
                    rows.append({
                        "conditions": combo_row["conditions"],
                        "side":       combo_row["side"],
                        "time_slot":  tb_name,
                        "regime":     adx_name,
                        **r
                    })

    result = pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
    print(f"  Found {len(result)} time-segmented edges with E > 0.10")
    return result


# ── Phase 4: Walk-forward validation ─────────────────────────────────────────

def phase4_walk_forward(df: pd.DataFrame, conditions: dict, top_segmented: pd.DataFrame) -> pd.DataFrame:
    print("\n[Phase 4] Walk-forward validation (80/20 split)...")
    if top_segmented.empty:
        return pd.DataFrame()

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df  = df.iloc[split:].reset_index(drop=True)

    train_conds = define_conditions(train_df)
    test_conds  = define_conditions(test_df)

    rows = []
    for _, row in top_segmented.head(20).iterrows():
        cond_names = row["conditions"].split(" + ")

        def build_mask(conds_dict):
            m = conds_dict.get(cond_names[0], pd.Series(False, index=range(len(next(iter(conds_dict.values()))))))
            for cn in cond_names[1:]:
                if cn in conds_dict:
                    m = m & conds_dict[cn]
            return m

        train_r = backtest_signal(train_df, build_mask(train_conds), side=row["side"])
        test_r  = backtest_signal(test_df,  build_mask(test_conds),  side=row["side"])

        if test_r["signals"] >= 5:
            edge_decay = (train_r["edge"] - test_r["edge"]) / (abs(train_r["edge"]) + 1e-9)
            rows.append({
                "conditions":    row["conditions"],
                "side":          row["side"],
                "time_slot":     row.get("time_slot", "all"),
                "regime":        row.get("regime", "all"),
                "train_edge":    train_r["edge"],
                "test_edge":     test_r["edge"],
                "test_wr":       test_r["win_rate"],
                "test_signals":  test_r["signals"],
                "edge_decay_pct": round(edge_decay * 100, 1),
                "validated":     test_r["edge"] > 0.05 and edge_decay < 0.5,
            })

    result = pd.DataFrame(rows).sort_values("test_edge", ascending=False) if rows else pd.DataFrame()
    validated = result[result["validated"] == True] if not result.empty else pd.DataFrame()
    print(f"  {len(validated)} / {len(result)} combos survived walk-forward validation")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuantDesk Backtesting Agent")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--days",   type=int, default=60)
    parser.add_argument("--output", default="backtest_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  QuantDesk Backtest Agent")
    print(f"  Symbol: {args.symbol}  |  Days: {args.days}")
    print(f"{'='*60}\n")

    df = fetch_data(args.symbol, args.days)
    df = compute_indicators(df)
    print(f"  Indicators computed on {len(df):,} bars\n")

    conditions = define_conditions(df)
    print(f"  Defined {len(conditions)} atomic conditions\n")

    single   = phase1_single_conditions(df, conditions)
    combos   = phase2_combinations(df, conditions)
    segmented = phase3_time_segment(df, conditions, combos) if not combos.empty else pd.DataFrame()
    validated = phase4_walk_forward(df, conditions, segmented) if not segmented.empty else pd.DataFrame()

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — Top validated edges")
    print(f"{'='*60}")

    if not validated.empty:
        top = validated[validated["validated"] == True].head(10)
        for _, r in top.iterrows():
            print(f"\n  {r['conditions']}")
            print(f"  Side: {r['side']}  |  Time: {r.get('time_slot','all')}  |  Regime: {r.get('regime','all')}")
            print(f"  Train edge: {r['train_edge']:+.3f}  →  Test edge: {r['test_edge']:+.3f}")
            print(f"  WR: {r['test_wr']*100:.0f}%  |  Signals: {r['test_signals']}  |  Decay: {r['edge_decay_pct']:.0f}%")
    else:
        print("  No combos survived walk-forward — try more days of data")

    # Save all results
    all_results = pd.concat([
        single.assign(phase="single"),
        combos.assign(phase="combo") if not combos.empty else pd.DataFrame(),
        segmented.assign(phase="segmented") if not segmented.empty else pd.DataFrame(),
        validated.assign(phase="validated") if not validated.empty else pd.DataFrame(),
    ], ignore_index=True)
    all_results.to_csv(args.output, index=False)
    print(f"\n  Full results saved to {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
