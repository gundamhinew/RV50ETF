# src/features/make_feature_table.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rv(close: pd.Series, window: int) -> pd.Series:
    """
    Realized Volatility (annualization not needed for signal; keep raw scale).
    rv_t = std( logret_{t-window+1:t} )
    """
    logret = np.log(close).diff()
    return logret.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)


def range_bound_flag(close: pd.Series, lookback: int = 20, band: float = 0.04) -> pd.Series:
    """
    横盘：过去 lookback 天的价格区间相对中位数很小
    range_ratio = (max-min)/median
    """
    roll_max = close.rolling(lookback, min_periods=lookback).max()
    roll_min = close.rolling(lookback, min_periods=lookback).min()
    roll_med = close.rolling(lookback, min_periods=lookback).median()
    rr = (roll_max - roll_min) / roll_med
    return rr < band


def low_vol_flag(rv_slow: pd.Series, percentile_window: int = 252, q: float = 0.2) -> pd.Series:
    """
    相对低波：rv_slow 处于过去 percentile_window 的低分位（默认20%分位以下）
    """
    thresh = rv_slow.rolling(percentile_window, min_periods=max(60, percentile_window // 3)).quantile(q)
    return rv_slow <= thresh


def pressure_build_flag(pressure_z: pd.Series, z_th: float = 1.0, sustain: int = 3) -> pd.Series:
    """
    压力堆积：pressure_z 连续 >= z_th sustain 天
    """
    cond = pressure_z >= z_th
    return cond.rolling(sustain, min_periods=sustain).sum() >= sustain


def make_feature_table(symbol: str, universe: str, refresh: bool = False) -> pd.DataFrame:
    """
    Inputs:
      data/processed/{symbol}_daily.parquet
      data/processed/{universe}_pressure_main.parquet

    Output (DataFrame):
      date, close, rv_5, rv_20, low_vol, range_bound, pressure_build
    """
    root = project_root()

    etf_path = root / "data" / "processed" / f"{symbol}_daily.parquet"
    pres_path = root / "data" / "processed" / f"{universe}_pressure_main.parquet"

    if not etf_path.exists():
        raise FileNotFoundError(f"Missing ETF daily: {etf_path}")
    if not pres_path.exists():
        raise FileNotFoundError(f"Missing pressure: {pres_path}")

    etf = pd.read_parquet(etf_path)
    pres = pd.read_parquet(pres_path)

    # normalize
    etf["date"] = pd.to_datetime(etf["date"])
    pres["date"] = pd.to_datetime(pres["date"])
    etf = etf.sort_values("date").reset_index(drop=True)
    pres = pres.sort_values("date").reset_index(drop=True)

    # merge on date
    df = etf.merge(pres, on="date", how="inner")

    if "close" not in df.columns:
        raise KeyError(f"'close' not in ETF daily columns: {df.columns.tolist()}")

    df["rv_5"] = rv(df["close"], 5)
    df["rv_20"] = rv(df["close"], 20)

    df["low_vol"] = low_vol_flag(df["rv_20"])
    df["range_bound"] = range_bound_flag(df["close"])

    # use pressure_main_z60 as standardized pressure regime
    if "pressure_main_z60" not in df.columns:
        raise KeyError(f"'pressure_main_z60' not in pressure columns: {df.columns.tolist()}")

    df["pressure_build"] = pressure_build_flag(df["pressure_main_z60"], z_th=1.0, sustain=3)

    out = df[[
        "date", "close",
        "rv_5", "rv_20",
        "low_vol", "range_bound", "pressure_build",
        "pressure_main", "pressure_main_20", "pressure_main_z60",
    ]].copy()

    return out


def save_states(df: pd.DataFrame, symbol: str, universe: str) -> Path:
    root = project_root()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{universe}_states.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build states feature table for ETF+universe pressure.")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--universe", type=str, required=True)
    args = parser.parse_args()

    df = make_feature_table(symbol=args.symbol, universe=args.universe, refresh=False)
    path = save_states(df, symbol=args.symbol, universe=args.universe)
    print(f"Saved: {path}  rows={len(df)}  range={df['date'].min().date()}->{df['date'].max().date()}")
    print(df.tail(5))


if __name__ == "__main__":
    main()
