from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


def forward_rv_change(rv: pd.Series, h: int) -> pd.Series:
    return rv.shift(-h) - rv


def breakout_within_h(close: pd.Series, h: int, lookback: int = 20) -> pd.Series:
    """
    Whether price makes a new lookback-day high within next h days.
    """
    future_max = close.shift(-1).rolling(h, min_periods=1).max()
    past_high = close.rolling(lookback, min_periods=lookback).max()
    return future_max > past_high


def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append({
            "metric": c,
            "mean": s.mean(),
            "median": s.median(),
            "positive_ratio": (s > 0).mean()
        })
    return pd.DataFrame(rows)


def main() -> None:
    root = project_root()
    states_path = root / "data" / "processed" / "rv50_states.parquet"
    if not states_path.exists():
        raise FileNotFoundError(f"States file not found: {states_path}")

    df = pd.read_parquet(states_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    horizons = [5, 10, 20]

    # Build forward metrics
    for h in horizons:
        df[f"fwd_ret_{h}d"] = forward_log_return(df["close"], h)
        df[f"fwd_drv_{h}d"] = forward_rv_change(df["rv_20"], h)
        df[f"breakout_{h}d"] = breakout_within_h(df["close"], h)

    event = df[df["pressure_build"] == True]
    nonev = df[df["pressure_build"] == False]

    metrics = []
    for h in horizons:
        metrics.extend([
            f"fwd_ret_{h}d",
            f"fwd_drv_{h}d",
            f"breakout_{h}d",
        ])

    event_summary = summarize(event, metrics)
    event_summary["group"] = "event"

    nonev_summary = summarize(nonev, metrics)
    nonev_summary["group"] = "non_event"

    out = pd.concat([event_summary, nonev_summary], ignore_index=True)

    out_path = root / "data" / "processed" / "rv50_event_study.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print("Event count:", len(event))
    print("\n=== Event days summary ===")
    print(event_summary)
    print("\n=== Non-event days summary ===")
    print(nonev_summary)


if __name__ == "__main__":
    main()
