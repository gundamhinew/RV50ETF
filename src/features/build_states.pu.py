from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rolling_quantile(s: pd.Series, window: int, q: float, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(30, window // 4)
    return s.rolling(window=window, min_periods=min_periods).quantile(q)


def sustained_indicator(ind: pd.Series, k: int, alpha: float) -> pd.Series:
    """
    ind: boolean series
    return: boolean series, True if last k days have at least ceil(alpha*k) Trues
    """
    need = int(np.ceil(alpha * k))
    return ind.rolling(window=k, min_periods=k).sum() >= need


def main(
    symbol: str = "588080",
    W: int = 252,          # rolling window for percentile thresholds
    q_vol: float = 0.20,   # low vol percentile
    q_range: float = 0.30, # low range percentile (range-bound)
    K_vol: int = 15,
    alpha_vol: float = 0.80,
    K_range: int = 15,
    alpha_range: float = 0.80,
    L_sum: int = 20,       # cumulative pressure window
    K_pos: int = 20,       # persistence window for positive pressure
    alpha_pos: float = 0.60
) -> None:
    root = project_root()

    ft_path = root / "data" / "processed" / "rv50_feature_table.parquet"
    if not ft_path.exists():
        raise FileNotFoundError(f"Feature table not found: {ft_path}")

    df = pd.read_parquet(ft_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Required columns
    required = ["close", "rv_20", "range_40", "pressure_main"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature table: {missing}. Got columns: {df.columns.tolist()}")

    # -----------------------
    # LowVol: relative, percentile, sustained
    # -----------------------
    vol_q = rolling_quantile(df["rv_20"], window=W, q=q_vol)
    ind_low_vol_day = df["rv_20"] <= vol_q
    df["low_vol"] = sustained_indicator(ind_low_vol_day.astype(int), k=K_vol, alpha=alpha_vol)

    # -----------------------
    # RangeBound: relative, percentile, sustained
    # -----------------------
    range_q = rolling_quantile(df["range_40"], window=W, q=q_range)
    ind_range_day = df["range_40"] <= range_q
    df["range_bound"] = sustained_indicator(ind_range_day.astype(int), k=K_range, alpha=alpha_range)

    # -----------------------
    # PressureBuild: sustained + cumulative
    # -----------------------
    df["pressure_sum_L"] = df["pressure_main"].rolling(window=L_sum, min_periods=L_sum).sum()

    pos_day = (df["pressure_main"] > 0).astype(int)
    df["pressure_pos_cnt"] = pos_day.rolling(window=K_pos, min_periods=K_pos).sum()

    need_pos = int(np.ceil(alpha_pos * K_pos))
    df["pressure_build"] = (df["pressure_sum_L"] > 0) & (df["pressure_pos_cnt"] >= need_pos)

    # -----------------------
    # Combined state
    # -----------------------
    df["state_ready"] = df["low_vol"] & df["range_bound"] & df["pressure_build"]

    # Output columns: keep minimal + debugging thresholds
    out = df[[
        "date", "close",
        "rv_20", "range_40", "pressure_main",
        "low_vol", "range_bound", "pressure_build", "state_ready",
        "pressure_sum_L", "pressure_pos_cnt"
    ]].copy()

    out_path = root / "data" / "processed" / "rv50_states.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out)}  Date range: {out['date'].min().date()} -> {out['date'].max().date()}")
    print("State counts:")
    print(out[["low_vol", "range_bound", "pressure_build", "state_ready"]].sum(numeric_only=True))
    print(out.tail(10))


if __name__ == "__main__":
    main()
