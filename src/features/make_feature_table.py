from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Project root helper
# =========================
def project_root() -> Path:
    # make_feature_table.py is under RV50ETF/src/features/
    # parents[0]=features, parents[1]=src, parents[2]=RV50ETF
    return Path(__file__).resolve().parents[2]


# =========================
# Load datasets (from RV50ETF/data/processed)
# =========================
def load_etf_daily(symbol: str = "588080") -> pd.DataFrame:
    root = project_root()
    path = root / "data" / "processed" / f"{symbol}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ETF daily file not found: {path}")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"ETF daily file missing 'date' column: {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_pressure() -> pd.DataFrame:
    root = project_root()
    path = root / "data" / "processed" / "kc50_pressure_main.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pressure file not found: {path}")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"Pressure file missing 'date' column: {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================
# Feature engineering
# =========================
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build return / realized volatility / range-bound features on merged dataset.
    Assumes df includes: date, open, high, low, close (at least close).
    """
    out = df.copy()

    # --- Basic returns
    out["ret_1d"] = np.log(out["close"]).diff()

    # --- Realized volatility proxies (std of log returns)
    out["rv_20"] = out["ret_1d"].rolling(20, min_periods=10).std()
    out["rv_60"] = out["ret_1d"].rolling(60, min_periods=30).std()

    # --- Range-bound proxy: rolling (max-min)/mean on close
    w = 40
    roll_max = out["close"].rolling(w, min_periods=20).max()
    roll_min = out["close"].rolling(w, min_periods=20).min()
    roll_mean = out["close"].rolling(w, min_periods=20).mean()
    out["range_40"] = (roll_max - roll_min) / roll_mean

    # --- ATR (Average True Range)
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)
    out["atr_20"] = tr.rolling(20, min_periods=10).mean()

    return out


# =========================
# Main pipeline
# =========================
def main(symbol: str = "588080") -> None:
    root = project_root()
    out_path = root / "data" / "processed" / "rv50_feature_table.parquet"

    etf = load_etf_daily(symbol)
    pressure = load_pressure()

    # Inner join: keep only overlap dates (pressure history is short)
    merged = pd.merge(etf, pressure, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Compute features
    merged = calc_features(merged)

    # Save
    merged.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(merged)}  Date range: {merged['date'].min().date()} -> {merged['date'].max().date()}")
    print("Columns:", merged.columns.tolist())
    print(merged.tail(5))


if __name__ == "__main__":
    main("588080")
