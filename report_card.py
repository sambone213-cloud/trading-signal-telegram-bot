"""
report_card.py — EOD self-evaluation
────────────────────────────────────
Replays each day's signals through the SAME pipeline the live bot uses
(prior-day warmup bars, _compute_indicators, run_all_strategies, 2xATR
dedup per strategy/side) and scores every signal to its exit: TP/SL touch
intrabar, strategy exit condition, or end of day.

The agent calls build_report() at 16:05 ET and sends it to Telegram —
today's signals in detail plus a running multi-day track record per
strategy, compared against each strategy's backtest win rate.

Standalone:  python report_card.py [SYMBOL] [DAYS]
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import pandas as pd

# Backtest / walk-forward win rates each strategy shipped with — the bar
# the live track record is judged against.
EXPECTED_WR = {
    "Power Hour Dip":   0.74,
    "Keltner Bounce":   0.64,
    "Momentum Flip":    0.62,
    "BB+ADX Reversal":  0.55,
    "Trend Breakout":   0.54,
    "VWAP Reclaim":     0.52,
    "ORB Breakout":     0.50,
    "Opening Drive":    0.50,
    "Oversold Dip Buy": 0.49,
    "Lunch VWAP Hold":  0.49,
}

SHORT_NAMES = {
    "Power Hour Dip": "PowerHour", "Keltner Bounce": "Keltner",
    "Momentum Flip": "MomFlip",    "BB+ADX Reversal": "BB+ADX",
    "Trend Breakout": "TrendBrk",  "VWAP Reclaim": "VWAPRecl",
    "ORB Breakout": "ORB",         "Opening Drive": "OpenDrive",
    "Oversold Dip Buy": "DipBuy",  "Lunch VWAP Hold": "LunchVWAP",
}


def _fetch(symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    hist = yf.Ticker(symbol).history(period=f"{days + 2}d", interval="1m",
                                     auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()
    hist.index = hist.index.tz_convert("UTC")
    df = pd.DataFrame({
        "datetime": hist.index,
        "open": hist["Open"].values, "high": hist["High"].values,
        "low": hist["Low"].values, "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    }).sort_values("datetime").reset_index(drop=True)
    if len(df) > 1 and df["volume"].iloc[-1] == 0:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def replay_day(full: pd.DataFrame, dates: list, di: int) -> list:
    """Replay one trading date with prior-day warmup (mirrors live 2d fetch)."""
    from trading_alert_agent import _compute_indicators
    from trade_strategies import run_all_strategies, check_exit_conditions

    date = dates[di]
    day_dates = full["datetime"].dt.date  # market-hours bars: UTC date == ET date
    day_idx = full.index[day_dates == date]
    if len(day_idx) < 60:
        return []
    warmup_start = full.index[day_dates == dates[di - 1]][0] if di > 0 else day_idx[0]

    open_trades, closed, last_fire = [], [], {}

    for i in range(day_idx[0], day_idx[-1] + 1):
        window = _compute_indicators(full.iloc[warmup_start:i + 1])
        bar = window.iloc[-1]
        price = float(bar["close"]); hi = float(bar["high"]); lo = float(bar["low"])
        et = window["datetime"].iloc[-1].tz_convert("America/New_York").strftime("%H:%M")

        # manage open trades (skip the entry bar itself)
        still = []
        for t in open_trades:
            if i == t["entry_bar"]:
                still.append(t); continue
            exit_px, cause = None, None
            if t["side"] == "long":
                if lo <= t["sl"]:   exit_px, cause = t["sl"], "SL"
                elif hi >= t["tp"]: exit_px, cause = t["tp"], "TP"
            else:
                if hi >= t["sl"]:   exit_px, cause = t["sl"], "SL"
                elif lo <= t["tp"]: exit_px, cause = t["tp"], "TP"
            if exit_px is None:
                hint = check_exit_conditions(window, t["side"], t["strategy"],
                                             i - t["entry_bar"])
                if hint:
                    exit_px, cause = price, hint
            if exit_px is None:
                still.append(t)
            else:
                pnl = (exit_px - t["entry_price"]) if t["side"] == "long" \
                      else (t["entry_price"] - exit_px)
                closed.append({**t, "exit_et": et, "exit_price": exit_px,
                               "cause": cause, "pnl": pnl, "date": date})
        open_trades = still

        # new signals — every one tracked (no trade cap: this measures the
        # strategies, not the risk envelope)
        sigs = run_all_strategies(window, notify=False)
        if sigs:
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

    eod_px = float(full["close"].iloc[day_idx[-1]])
    for t in open_trades:
        pnl = (eod_px - t["entry_price"]) if t["side"] == "long" \
              else (t["entry_price"] - eod_px)
        closed.append({**t, "exit_et": "EOD", "exit_price": eod_px,
                       "cause": "EOD", "pnl": pnl, "date": date})
    closed.sort(key=lambda t: t["entry_et"])
    return closed


def build_report(symbol: str = "SPY", days: int = 5) -> str:
    full = _fetch(symbol, days)
    if full.empty:
        return f"📊 Report card: no {symbol} data available"

    all_dates = sorted(full["datetime"].dt.date.unique())
    dates = all_dates[-days:] if len(all_dates) > days else all_dates

    all_trades = []
    for di in range(len(dates)):
        if di == 0 and len(all_dates) > len(dates):
            # ensure warmup day exists before the first scored day
            dates_w = [all_dates[all_dates.index(dates[0]) - 1]] + dates
            all_trades += replay_day(full, dates_w, 1)
            continue
        all_trades += replay_day(full, dates, di)

    today = dates[-1]
    today_trades = [t for t in all_trades if t["date"] == today]

    day_dates = full["datetime"].dt.date
    day_df = full[day_dates == today]
    open_p, close_p = float(day_df["open"].iloc[0]), float(day_df["close"].iloc[-1])
    day_pct = (close_p / open_p - 1) * 100

    lines = [f"📊 <b>EOD REPORT CARD — {symbol}</b>  {today.strftime('%a %b %d')}",
             f"Day {day_pct:+.2f}%  ·  range "
             f"${float(day_df['low'].min()):.2f}–${float(day_df['high'].max()):.2f}",
             "──────────────────────"]

    if not today_trades:
        lines.append("No signals fired today.")
    else:
        wins = sum(1 for t in today_trades if t["pnl"] > 0)
        net = sum(t["pnl"] for t in today_trades)
        lines.append(f"<b>Today: {len(today_trades)} signals · "
                     f"{wins}/{len(today_trades)} wins · {net:+.2f} pts</b>")
        rows = []
        for t in today_trades:
            nm = SHORT_NAMES.get(t["strategy"], t["strategy"][:9])
            sd = "L" if t["side"] == "long" else "S"
            cause = t["cause"] if len(str(t["cause"])) <= 18 else str(t["cause"])[:18]
            rows.append(f"{t['entry_et']} {nm:<9} {sd} {t['conf']:.2f} "
                        f"{t['pnl']:+5.2f} {cause}")
        lines.append("<pre>" + "\n".join(rows) + "</pre>")

    # running track record across all replayed days
    lines.append(f"──────────────────────")
    lines.append(f"<b>Track record ({len(dates)} session{'s' if len(dates) > 1 else ''}):</b>")
    by = {}
    for t in all_trades:
        d = by.setdefault(t["strategy"], {"n": 0, "w": 0, "pnl": 0.0})
        d["n"] += 1; d["w"] += t["pnl"] > 0; d["pnl"] += t["pnl"]
    rows = [f"{'strategy':<10}{'n':>3}{'win%':>6}{'exp%':>6}{'pts':>7}"]
    flags = []
    for k, d in sorted(by.items(), key=lambda x: -x[1]["pnl"]):
        wr = d["w"] / d["n"] if d["n"] else 0
        exp = EXPECTED_WR.get(k)
        nm = SHORT_NAMES.get(k, k[:9])
        rows.append(f"{nm:<10}{d['n']:>3}{wr * 100:>6.0f}"
                    f"{(exp * 100 if exp else 0):>6.0f}{d['pnl']:>+7.2f}")
        # flag only with a real sample and a big shortfall vs expectation
        if exp and d["n"] >= 8 and wr < exp - 0.20:
            flags.append(k)
    lines.append("<pre>" + "\n".join(rows) + "</pre>")

    if flags:
        lines.append("⚠️ Underperforming vs backtest (n≥8, >20pt shortfall): "
                     + ", ".join(flags))
    else:
        lines.append("✅ No strategy outside expected range yet "
                     "(needs n≥8 and >20pt WR shortfall to flag)")

    text = "\n".join(lines)
    return text[:4000]  # Telegram message limit safety


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = build_report(sym, n)
    # strip HTML tags for console preview
    import re
    print(re.sub(r"</?(b|pre)>", "", out))
