"""
backtest_agent.py  —  QuantDesk Exhaustive Backtester
──────────────────────────────────────────────────────
Data   : Alpaca (1-min bars, real OHLCV via IEX feed)
Search : Phase 1 single conditions -> Phase 2 all 2-way combos of top 40
         -> Phase 3 all 3-way combos of top 25 -> Phase 4 time+regime segment
         -> Phase 5 rolling walk-forward (3 windows)

Usage:
    python backtest_agent.py
    python backtest_agent.py --symbol SPY --days 500 --min-freq 1.0
"""

import argparse
import itertools
import warnings
import datetime
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Load .env from this file's directory or parent ────────────────────────────

def _load_dotenv():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [here, os.path.dirname(here)]:
        env_path = os.path.join(folder, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            return

_load_dotenv()

# ── Alpaca data ───────────────────────────────────────────────────────────────

def _raw_chunk_to_df(raw) -> pd.DataFrame:
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)
    raw.index = raw.index.tz_convert("UTC")
    return pd.DataFrame({
        "datetime": raw.index,
        "open":   raw["open"].values,
        "high":   raw["high"].values,
        "low":    raw["low"].values,
        "close":  raw["close"].values,
        "volume": raw["volume"].values,
    })


def fetch_alpaca(symbol: str = "SPY", years: float = 2) -> pd.DataFrame:
    """
    Fetch in 1-month chunks so we see progress and avoid silent hangs.
    Alpaca free/paper tier: IEX feed, ~2 years of 1-min history available.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        print("  alpaca-py not installed. Run: pip install alpaca-py")
        sys.exit(1)

    api_key    = os.environ.get("ALPACA_API_KEY", "")
    # accept both ALPACA_API_SECRET and ALPACA_SECRET_KEY (Railway vs local .env)
    api_secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        print("  Alpaca keys not found. Set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    client     = StockHistoricalDataClient(api_key, api_secret)
    now        = datetime.datetime.now(datetime.timezone.utc)
    total_days = int(365 * years)
    CHUNK_DAYS = 30

    chunks     = []
    chunk_end  = now
    fetched    = 0

    print(f"  Fetching {years:.1f}y of {symbol} 1-min bars from Alpaca ({total_days//CHUNK_DAYS} chunks)...")

    while fetched < total_days:
        chunk_start = chunk_end - datetime.timedelta(days=CHUNK_DAYS)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=chunk_start,
                end=chunk_end,
                feed="iex",
            )
            bars = client.get_stock_bars(req)
            if bars.df is not None and not bars.df.empty:
                chunk_df = _raw_chunk_to_df(bars.df)
                chunks.append(chunk_df)
                print(f"    {chunk_start.date()} -> {chunk_end.date()}: {len(chunk_df):,} bars")
            else:
                print(f"    {chunk_start.date()} -> {chunk_end.date()}: no data")
        except Exception as e:
            print(f"    {chunk_start.date()} -> {chunk_end.date()}: skipped ({e})")

        chunk_end  = chunk_start
        fetched   += CHUNK_DAYS

    if not chunks:
        raise RuntimeError("No data returned from Alpaca — check API keys and account type")

    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    # Regular market hours only: 9:30-16:00 ET = 13:30-20:00 UTC
    h = df["datetime"].dt.hour
    m = df["datetime"].dt.minute
    market = ((h == 13) & (m >= 30)) | ((h >= 14) & (h < 20))
    df = df[market].reset_index(drop=True)

    print(f"  Total: {len(df):,} market-hours bars "
          f"({df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()})")
    return df


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    v   = df["volume"]
    n   = len(df)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for span in [5, 9, 13, 21, 34, 50, 89, 200]:
        df[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()

    # ── SMAs ──────────────────────────────────────────────────────────────────
    for w in [10, 20, 50]:
        df[f"sma{w}"] = c.rolling(w).mean()

    # ── RSI-14 ────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── Stochastic %K/%D (14,3) ───────────────────────────────────────────────
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Williams %R (14) ──────────────────────────────────────────────────────
    df["willr"] = -100 * (hi14 - c) / (hi14 - lo14).replace(0, np.nan)

    # ── CCI-20 ────────────────────────────────────────────────────────────────
    tp = (h + l + c) / 3
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────────
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(0, np.nan)
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # ── Keltner Channel (20, 2xATR) ───────────────────────────────────────────
    prev_c = c.shift(1)
    tr     = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["atr"]       = tr.rolling(14).mean()
    df["atr_pct"]   = df["atr"] / c.replace(0, np.nan)
    df["kc_upper"]  = sma20 + 2 * df["atr"]
    df["kc_lower"]  = sma20 - 2 * df["atr"]

    # ── Squeeze (BB inside KC) ────────────────────────────────────────────────
    df["squeeze"]   = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    df["bb_squeeze"]= df["bb_width"] < df["bb_width"].rolling(20).mean() * 0.7

    # ── ADX-14 ────────────────────────────────────────────────────────────────
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14    = tr.rolling(14).mean()
    df["plus_di"]  = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    df["minus_di"] = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / \
         (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    df["adx"] = dx.rolling(14).mean()

    # ── VWAP ──────────────────────────────────────────────────────────────────
    # Reset each trading day
    df["date"] = df["datetime"].dt.date
    df["vwap"] = (
        (c * v).groupby(df["date"]).cumsum() /
        v.groupby(df["date"]).cumsum()
    )
    df.drop(columns=["date"], inplace=True)

    # ── VWMA-9 / VWMA-21 ──────────────────────────────────────────────────────
    df["vwma9"]  = (c * v).rolling(9).sum()  / v.rolling(9).sum()
    df["vwma21"] = (c * v).rolling(21).sum() / v.rolling(21).sum()

    # ── MFI-14 (Money Flow Index) ─────────────────────────────────────────────
    mf  = tp * v
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df["mfi"] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

    # ── OBV trend ─────────────────────────────────────────────────────────────
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"] = obv - obv.rolling(10).mean()   # positive = rising OBV

    # ── Rate of Change (ROC) ──────────────────────────────────────────────────
    df["roc5"]  = c.pct_change(5)  * 100
    df["roc10"] = c.pct_change(10) * 100
    df["roc20"] = c.pct_change(20) * 100

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["mom10"] = c - c.shift(10)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"].replace(0, np.nan)
    df["vol_surge"] = df["vol_ratio"] > 1.5

    # ── Time features (ET = UTC-4 approx) ────────────────────────────────────
    utc_h = df["datetime"].dt.hour
    utc_m = df["datetime"].dt.minute
    et_h  = utc_h - 4
    et_m  = utc_m
    df["hour_et"]    = et_h
    df["minute_et"]  = et_m
    df["mins_since_open"] = (et_h - 9) * 60 + et_m - 30
    df["time_bucket"] = df["mins_since_open"] // 30   # 30-min buckets 0-12

    return df.dropna().reset_index(drop=True)


# ── Atomic conditions ─────────────────────────────────────────────────────────

def build_conditions(df: pd.DataFrame) -> Dict[str, pd.Series]:
    c = df["close"]
    b = {}

    # ── Trend / EMA ───────────────────────────────────────────────────────────
    b["ema9>21"]      = df["ema9"]  > df["ema21"]
    b["ema9<21"]      = df["ema9"]  < df["ema21"]
    b["ema21>50"]     = df["ema21"] > df["ema50"]
    b["ema21<50"]     = df["ema21"] < df["ema50"]
    b["ema50>200"]    = df["ema50"] > df["ema200"]
    b["price>ema9"]   = c > df["ema9"]
    b["price<ema9"]   = c < df["ema9"]
    b["price>ema21"]  = c > df["ema21"]
    b["price<ema21"]  = c < df["ema21"]
    b["price>ema50"]  = c > df["ema50"]
    b["ema9_xup"]     = (df["ema9"].shift(1) <= df["ema21"].shift(1)) & (df["ema9"] > df["ema21"])
    b["ema9_xdn"]     = (df["ema9"].shift(1) >= df["ema21"].shift(1)) & (df["ema9"] < df["ema21"])
    b["price>vwap"]   = c > df["vwap"]
    b["price<vwap"]   = c < df["vwap"]
    b["vwma9>21"]     = df["vwma9"] > df["vwma21"]
    b["vwma9<21"]     = df["vwma9"] < df["vwma21"]
    b["vwma_xup"]     = (df["vwma9"].shift(1) <= df["vwma21"].shift(1)) & (df["vwma9"] > df["vwma21"])
    b["vwma_xdn"]     = (df["vwma9"].shift(1) >= df["vwma21"].shift(1)) & (df["vwma9"] < df["vwma21"])

    # ── RSI ───────────────────────────────────────────────────────────────────
    b["rsi<25"]       = df["rsi"] < 25
    b["rsi<30"]       = df["rsi"] < 30
    b["rsi<35"]       = df["rsi"] < 35
    b["rsi<40"]       = df["rsi"] < 40
    b["rsi>60"]       = df["rsi"] > 60
    b["rsi>65"]       = df["rsi"] > 65
    b["rsi>70"]       = df["rsi"] > 70
    b["rsi>75"]       = df["rsi"] > 75
    b["rsi>78"]       = df["rsi"] > 78
    b["rsi_rising"]   = df["rsi"] > df["rsi"].shift(1)
    b["rsi_falling"]  = df["rsi"] < df["rsi"].shift(1)
    b["rsi50_xup"]    = (df["rsi"].shift(1) < 50) & (df["rsi"] >= 50)
    b["rsi50_xdn"]    = (df["rsi"].shift(1) > 50) & (df["rsi"] <= 50)

    # ── Stochastic ────────────────────────────────────────────────────────────
    b["stoch<20"]     = df["stoch_k"] < 20
    b["stoch>80"]     = df["stoch_k"] > 80
    b["stoch_xup"]    = (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"] > df["stoch_d"])
    b["stoch_xdn"]    = (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"] < df["stoch_d"])

    # ── Williams %R ───────────────────────────────────────────────────────────
    b["willr<-80"]    = df["willr"] < -80
    b["willr>-20"]    = df["willr"] > -20

    # ── CCI ───────────────────────────────────────────────────────────────────
    b["cci<-100"]     = df["cci"] < -100
    b["cci>100"]      = df["cci"] > 100
    b["cci>0"]        = df["cci"] > 0
    b["cci<0"]        = df["cci"] < 0

    # ── MACD ──────────────────────────────────────────────────────────────────
    b["macd>0"]       = df["macd"] > 0
    b["macd<0"]       = df["macd"] < 0
    b["macd_xup"]     = (df["macd"].shift(1) <= df["macd_signal"].shift(1)) & (df["macd"] > df["macd_signal"])
    b["macd_xdn"]     = (df["macd"].shift(1) >= df["macd_signal"].shift(1)) & (df["macd"] < df["macd_signal"])
    b["hist_expand"]  = df["macd_hist"].abs() > df["macd_hist"].shift(1).abs()
    b["hist_shrink"]  = df["macd_hist"].abs() < df["macd_hist"].shift(1).abs()

    # ── Bollinger / Keltner ───────────────────────────────────────────────────
    b["above_bbu"]    = c >= df["bb_upper"]
    b["below_bbl"]    = c <= df["bb_lower"]
    b["inside_bb"]    = (c < df["bb_upper"]) & (c > df["bb_lower"])
    b["bb_pct>80"]    = df["bb_pct"] > 0.8
    b["bb_pct<20"]    = df["bb_pct"] < 0.2
    b["bb_squeeze"]   = df["bb_squeeze"]
    b["squeeze_on"]   = df["squeeze"]
    b["above_kcu"]    = c > df["kc_upper"]
    b["below_kcl"]    = c < df["kc_lower"]

    # ── ADX / Trend strength ──────────────────────────────────────────────────
    b["adx<20"]       = df["adx"] < 20
    b["adx<25"]       = df["adx"] < 25
    b["adx>20"]       = df["adx"] > 20
    b["adx>25"]       = df["adx"] > 25
    b["adx>30"]       = df["adx"] > 30
    b["bullish_di"]   = df["plus_di"] > df["minus_di"]
    b["bearish_di"]   = df["plus_di"] < df["minus_di"]

    # ── Volume ────────────────────────────────────────────────────────────────
    b["vol>1.5x"]     = df["vol_ratio"] > 1.5
    b["vol>2x"]       = df["vol_ratio"] > 2.0
    b["vol_surge"]    = df["vol_surge"]
    b["obv_rising"]   = df["obv_slope"] > 0
    b["obv_falling"]  = df["obv_slope"] < 0

    # ── MFI ───────────────────────────────────────────────────────────────────
    b["mfi<20"]       = df["mfi"] < 20
    b["mfi>80"]       = df["mfi"] > 80

    # ── Momentum / ROC ────────────────────────────────────────────────────────
    b["roc5>0"]       = df["roc5"] > 0
    b["roc5<0"]       = df["roc5"] < 0
    b["roc10>0.5"]    = df["roc10"] > 0.5
    b["roc10<-0.5"]   = df["roc10"] < -0.5
    b["mom10>0"]      = df["mom10"] > 0
    b["mom10<0"]      = df["mom10"] < 0

    # ── Time of day ───────────────────────────────────────────────────────────
    b["opening_30"]   = df["mins_since_open"] < 30
    b["after_30m"]    = df["mins_since_open"] >= 30
    b["after_60m"]    = df["mins_since_open"] >= 60
    b["lunch"]        = (df["hour_et"] == 12)
    b["power_hour"]   = (df["hour_et"] == 15)
    b["morning"]      = (df["mins_since_open"] >= 30) & (df["hour_et"] < 12)
    b["afternoon"]    = (df["hour_et"] >= 13) & (df["hour_et"] < 15)

    return {k: v.astype(bool) for k, v in b.items()}


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    mask: pd.Series,
    side: str,
    hold_bars: int = 5,
    tp_atr: float = 2.5,
    sl_atr: float = 1.0,
) -> dict:
    """
    Fully vectorized backtest — no Python loops over bars.
    Uses numpy stride tricks to check TP/SL across hold_bars window.
    ~100x faster than the row-by-row version.
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = df["atr"].values
    sig   = mask.values.astype(bool)
    n     = len(close)

    # Entry indices: signal is True, not too close to end, atr > 0
    idxs = np.where(sig)[0]
    idxs = idxs[(idxs >= 50) & (idxs + hold_bars < n) & (atr[idxs] > 0)]

    # Deduplicate — skip entries within 3 bars of the previous
    if len(idxs) == 0:
        return {}
    gaps = np.diff(idxs, prepend=idxs[0] - 999)
    idxs = idxs[gaps >= 3]

    if len(idxs) < 15:
        return {}

    entry_price = close[idxs]
    entry_atr   = atr[idxs]

    if side == "long":
        tp_price = entry_price + tp_atr * entry_atr
        sl_price = entry_price - sl_atr * entry_atr
    else:
        tp_price = entry_price - tp_atr * entry_atr
        sl_price = entry_price + sl_atr * entry_atr

    # Build (n_entries, hold_bars) windows for high and low
    # Each row i = bars[idxs[i]+1 .. idxs[i]+hold_bars]
    row_idx = idxs[:, None] + np.arange(1, hold_bars + 1)   # shape (E, H)
    win_high = high[row_idx]   # (E, H)
    win_low  = low[row_idx]    # (E, H)
    win_close= close[idxs + hold_bars]  # exit close if no TP/SL

    if side == "long":
        tp_hit = win_high >= tp_price[:, None]   # (E, H)
        sl_hit = win_low  <= sl_price[:, None]
    else:
        tp_hit = win_low  <= tp_price[:, None]
        sl_hit = win_high >= sl_price[:, None]

    # For each entry find the first bar that hits TP or SL
    outcomes = np.full(len(idxs), np.nan)
    for b in range(hold_bars):
        undecided = np.isnan(outcomes)
        tp_now = undecided & tp_hit[:, b]
        sl_now = undecided & sl_hit[:, b] & ~tp_now
        outcomes[tp_now] =  tp_atr
        outcomes[sl_now] = -sl_atr

    # Remaining: use close-at-expiry
    still_open = np.isnan(outcomes)
    if side == "long":
        outcomes[still_open] = (win_close[still_open] - entry_price[still_open]) / entry_atr[still_open]
    else:
        outcomes[still_open] = (entry_price[still_open] - win_close[still_open]) / entry_atr[still_open]

    if len(outcomes) < 15:
        return {}

    wins   = outcomes[outcomes > 0]
    losses = outcomes[outcomes <= 0]
    edge   = float(np.mean(outcomes))
    wr     = len(wins) / len(outcomes)
    std    = float(np.std(outcomes)) or 1e-9
    return {
        "n":        len(outcomes),
        "wr":       round(wr, 3),
        "edge":     round(edge, 4),
        "sharpe":   round(edge / std * np.sqrt(252 * 390 / hold_bars), 2),
        "avg_win":  round(float(np.mean(wins))   if len(wins)   else 0, 3),
        "avg_loss": round(float(np.mean(losses)) if len(losses) else 0, 3),
        "pf":       round(-float(np.mean(wins)) / float(np.mean(losses)) if len(losses) and len(wins) else 0, 2),
    }


# ── Phase 1: Single condition scan ───────────────────────────────────────────

def phase1(df, conds):
    print("\n[Phase 1] Single conditions...")
    rows = []
    for name, mask in conds.items():
        for side in ["long", "short"]:
            r = backtest(df, mask, side)
            if r:
                rows.append({"label": name, "side": side, "combo": name, **r})
    result = pd.DataFrame(rows).sort_values("edge", ascending=False)
    print(f"  {len(conds)} conditions x 2 sides -> {len(result)} with >=15 signals")
    return result


# ── Phase 2: All 2-way combos of top N ───────────────────────────────────────

def phase2(df, conds, single_results, top_n=40):
    print(f"\n[Phase 2] 2-way combos (top {top_n} conditions)...")
    top = single_results[single_results["edge"] > 0]["label"].drop_duplicates().head(top_n).tolist()
    rows = []
    total = 0
    combos = list(itertools.combinations(top, 2))
    for c1, c2 in combos:
        mask = conds[c1] & conds[c2]
        for side in ["long", "short"]:
            r = backtest(df, mask, side)
            if r and r["edge"] > 0.04:
                rows.append({"label": f"{c1} & {c2}", "side": side,
                             "combo": f"{c1} & {c2}", **r})
        total += 1
        if total % 100 == 0:
            print(f"    {total}/{len(combos)} combos tested...", end="\r")
    result = pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
    print(f"  {total} combos -> {len(result)} with edge > 0.04        ")
    return result


# ── Phase 3: All 3-way combos of top N ───────────────────────────────────────

def phase3(df, conds, single_results, top_n=25):
    print(f"\n[Phase 3] 3-way combos (top {top_n} conditions)...")
    top = single_results[single_results["edge"] > 0]["label"].drop_duplicates().head(top_n).tolist()
    rows = []
    total = 0
    combos = list(itertools.combinations(top, 3))
    for c1, c2, c3 in combos:
        mask = conds[c1] & conds[c2] & conds[c3]
        for side in ["long", "short"]:
            r = backtest(df, mask, side)
            if r and r["edge"] > 0.07:
                rows.append({"label": f"{c1} & {c2} & {c3}", "side": side,
                             "combo": f"{c1} & {c2} & {c3}", **r})
        total += 1
        if total % 200 == 0:
            print(f"    {total}/{len(combos)} combos tested...", end="\r")
    result = pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
    print(f"  {total} combos -> {len(result)} with edge > 0.07        ")
    return result


# ── Phase 4: Time + regime segmentation ──────────────────────────────────────

def phase4_segment(df, conds, top_combos, top_n=30):
    print(f"\n[Phase 4] Time + regime segmentation...")
    if top_combos.empty:
        return pd.DataFrame()

    time_slots = {
        "open_30":   (df["mins_since_open"] >= 0)  & (df["mins_since_open"] < 30),
        "30-60m":    (df["mins_since_open"] >= 30) & (df["mins_since_open"] < 60),
        "60-90m":    (df["mins_since_open"] >= 60) & (df["mins_since_open"] < 90),
        "90-120m":   (df["mins_since_open"] >= 90) & (df["mins_since_open"] < 120),
        "lunch":      df["hour_et"] == 12,
        "afternoon": (df["hour_et"] >= 13) & (df["hour_et"] < 15),
        "power_hour": df["hour_et"] == 15,
    }
    regimes = {
        "trending":  df["adx"] > 25,
        "ranging":   df["adx"] < 20,
        "any":       pd.Series(True, index=df.index),
    }

    rows = []
    for _, row in top_combos.head(top_n).iterrows():
        parts = row["label"].split(" & ")

        def build(df_local):
            m = conds.get(parts[0], pd.Series(False, index=df_local.index))
            for p in parts[1:]:
                m = m & conds.get(p, pd.Series(True, index=df_local.index))
            return m

        base_mask = build(df)
        for ts_name, ts_mask in time_slots.items():
            for reg_name, reg_mask in regimes.items():
                m = base_mask & ts_mask & reg_mask
                r = backtest(df, m, row["side"])
                if r and r["edge"] > 0.06 and r["n"] >= 10:
                    rows.append({
                        "label":   row["label"],
                        "side":    row["side"],
                        "time":    ts_name,
                        "regime":  reg_name,
                        **r
                    })

    result = pd.DataFrame(rows).sort_values("edge", ascending=False) if rows else pd.DataFrame()
    print(f"  Found {len(result)} segmented setups with edge > 0.06")
    return result


# ── Phase 5: Rolling walk-forward ────────────────────────────────────────────

def _time_mask(d: pd.DataFrame, name) -> pd.Series:
    if name is None or (isinstance(name, float) and np.isnan(name)) or name == "all":
        return pd.Series(True, index=d.index)
    slots = {
        "open_30":   (d["mins_since_open"] >= 0)  & (d["mins_since_open"] < 30),
        "30-60m":    (d["mins_since_open"] >= 30) & (d["mins_since_open"] < 60),
        "60-90m":    (d["mins_since_open"] >= 60) & (d["mins_since_open"] < 90),
        "90-120m":   (d["mins_since_open"] >= 90) & (d["mins_since_open"] < 120),
        "lunch":      d["hour_et"] == 12,
        "afternoon": (d["hour_et"] >= 13) & (d["hour_et"] < 15),
        "power_hour": d["hour_et"] == 15,
    }
    return slots.get(name, pd.Series(True, index=d.index))


def _regime_mask(d: pd.DataFrame, name) -> pd.Series:
    if name is None or (isinstance(name, float) and np.isnan(name)) or name in ("all", "any"):
        return pd.Series(True, index=d.index)
    regimes = {
        "trending": d["adx"] > 25,
        "ranging":  d["adx"] < 20,
    }
    return regimes.get(name, pd.Series(True, index=d.index))


def phase5_walkforward(df, conds, candidates, n_windows=3, min_freq=0.0):
    """
    Walk-forward validate EVERY candidate (no top-20 cap).
    Conditions are precomputed once per window, then reused across all
    candidates — this is what makes validating 1000+ setups feasible.
    Segmented candidates keep their time/regime masks during validation.
    """
    print(f"\n[Phase 5] Rolling walk-forward ({n_windows} windows, {len(candidates)} candidates)...")
    if candidates.empty:
        return pd.DataFrame()

    total_bars = len(df)
    window     = total_bars // (n_windows + 1)

    # Precompute window slices + their condition dicts ONCE
    windows = []
    for w in range(n_windows):
        train_start = w * window
        train_end   = train_start + window
        test_start  = train_end
        test_end    = min(test_start + window // 2, total_bars)

        train_df = df.iloc[train_start:train_end].reset_index(drop=True)
        test_df  = df.iloc[test_start:test_end].reset_index(drop=True)
        windows.append((train_df, build_conditions(train_df),
                        test_df,  build_conditions(test_df)))

    rows  = []
    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        parts    = row["label"].split(" & ")
        t_name   = row.get("time", "all")
        r_name   = row.get("regime", "all")
        window_results = []

        for train_df, train_c, test_df, test_c in windows:

            def masked(c_dict, df_local):
                m = c_dict.get(parts[0], pd.Series(False, index=df_local.index))
                for p in parts[1:]:
                    m = m & c_dict.get(p, pd.Series(True, index=df_local.index))
                return m & _time_mask(df_local, t_name) & _regime_mask(df_local, r_name)

            tr = backtest(train_df, masked(train_c, train_df), row["side"])
            te = backtest(test_df,  masked(test_c,  test_df),  row["side"])
            if tr and te:
                test_days = max(len(test_df) / 390, 1e-9)
                window_results.append((tr["edge"], te["edge"], te["n"],
                                       te["wr"], te["n"] / test_days))

        if i % 100 == 0:
            print(f"    {i}/{total} candidates validated...", end="\r")

        if len(window_results) >= 2:
            avg_train = np.mean([x[0] for x in window_results])
            avg_test  = np.mean([x[1] for x in window_results])
            total_n   = sum(x[2] for x in window_results)
            avg_wr    = np.mean([x[3] for x in window_results])
            avg_freq  = np.mean([x[4] for x in window_results])
            decay     = (avg_train - avg_test) / (abs(avg_train) + 1e-9)
            rows.append({
                "label":        row["label"],
                "side":         row["side"],
                "time":         t_name if isinstance(t_name, str) else "all",
                "regime":       r_name if isinstance(r_name, str) else "all",
                "train_edge":   round(avg_train, 4),
                "test_edge":    round(avg_test, 4),
                "test_wr":      round(avg_wr, 3),
                "test_freq":    round(avg_freq, 2),
                "test_n":       total_n,
                "consistency":  round(avg_freq * avg_wr, 3),
                "decay_pct":    round(decay * 100, 1),
                "validated":    avg_test > 0.05 and decay < 0.60 and avg_freq >= min_freq,
            })

    result = pd.DataFrame(rows).sort_values("consistency", ascending=False) if rows else pd.DataFrame()
    if not result.empty:
        n_val = result["validated"].sum()
        print(f"  {n_val}/{len(result)} setups validated "
              f"(test edge > 0.05, decay < 60%, freq >= {min_freq:.1f}/day)")
    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(validated: pd.DataFrame, top: int = 25):
    print(f"\n{'='*70}")
    print(f"  TOP VALIDATED SETUPS  (ranked by consistency = freq x win-rate)")
    print(f"{'='*70}")
    good = validated[validated["validated"] == True].head(top) if not validated.empty else pd.DataFrame()
    if good.empty:
        print("  None passed walk-forward validation")
        return
    for i, (_, r) in enumerate(good.iterrows(), 1):
        print(f"\n  #{i}  {r['label']}")
        print(f"       Side: {r['side'].upper()}  |  Time: {r.get('time','all')}  |  Regime: {r.get('regime','all')}")
        print(f"       Train: {r['train_edge']:+.3f}  ->  Test: {r['test_edge']:+.3f}  "
              f"|  WR {r['test_wr']*100:.0f}%  freq={r['test_freq']:.2f}/day  "
              f"|  Consistency {r['consistency']:.2f}  |  Decay {r['decay_pct']:.0f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="SPY")
    parser.add_argument("--years",    type=float, default=None)
    parser.add_argument("--days",     type=int,   default=None)
    parser.add_argument("--min-freq", type=float, default=0.0,
                        help="Minimum average trades per trading day (filters final output)")
    parser.add_argument("--output",   default="backtest_results.csv")
    args = parser.parse_args()

    # Resolve years from --days if provided; default to 2 years
    if args.days is not None:
        years = args.days / 365.0
        label = f"{args.days}d"
    else:
        years = args.years if args.years is not None else 2
        label = f"{years}y"

    print(f"\n{'='*70}")
    print(f"  QuantDesk Exhaustive Backtest  —  {args.symbol}  {label}  1-min")
    if args.min_freq > 0:
        print(f"  Min-freq filter: >= {args.min_freq:.2f} trades/trading-day")
    print(f"{'='*70}\n")

    df = fetch_alpaca(args.symbol, years)
    df = compute_indicators(df)
    print(f"  Indicators computed on {len(df):,} bars\n")

    conds = build_conditions(df)
    print(f"  Defined {len(conds)} atomic conditions\n")

    single   = phase1(df, conds)
    two_way  = phase2(df, conds, single, top_n=40)
    three_way= phase3(df, conds, single, top_n=25)

    all_combos = pd.concat([two_way, three_way], ignore_index=True) \
                   .sort_values("edge", ascending=False) \
                   .drop_duplicates("label") if not two_way.empty else three_way

    segmented  = phase4_segment(df, conds, all_combos, top_n=30)

    trading_days = int(len(df) / 390)  # ~390 1-min bars per trading day

    # ── Candidate pool: EVERYTHING that fires often enough ────────────────────
    # Singles + 2-way + 3-way + segmented, deduped, gated by min-freq up front.
    pool = pd.concat([
        single, all_combos,
        segmented if not segmented.empty else pd.DataFrame(),
    ], ignore_index=True)
    pool = pool.drop_duplicates(subset=["label", "side", "time", "regime"]
                                if "time" in pool.columns else ["label", "side"])
    if args.min_freq > 0 and trading_days > 0 and "n" in pool.columns:
        pool = pool[pool["n"] / trading_days >= args.min_freq]
    pool = pool.sort_values("edge", ascending=False).reset_index(drop=True)
    print(f"\n  Candidate pool after freq gate (>= {args.min_freq:.1f}/day): {len(pool)} setups")

    validated = phase5_walkforward(df, conds, pool, min_freq=args.min_freq)
    print_report(validated)

    # Save all phases
    all_results = pd.concat([
        single.assign(phase="single"),
        two_way.assign(phase="2way")      if not two_way.empty   else pd.DataFrame(),
        three_way.assign(phase="3way")    if not three_way.empty else pd.DataFrame(),
        segmented.assign(phase="segment") if not segmented.empty else pd.DataFrame(),
        validated.assign(phase="validated") if not validated.empty else pd.DataFrame(),
    ], ignore_index=True)
    all_results.to_csv(args.output, index=False)
    print(f"\n  Full results -> {args.output}")

    # Recommended portfolio: validated setups only, ranked by consistency
    if not validated.empty:
        rec = validated[validated["validated"] == True].copy()
        if not rec.empty:
            rec["note"] = rec.apply(
                lambda r: f"{r['side']} | {r['time']}/{r['regime']} | "
                          f"WR {r['test_wr']*100:.0f}% at {r['test_freq']:.1f}/day", axis=1)
            rec.to_csv("backtest_recommended.csv", index=False)
            print(f"  Recommended portfolio ({len(rec)} setups) -> backtest_recommended.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
