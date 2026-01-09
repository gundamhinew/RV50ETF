from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_setup_flag(pressure_build: pd.Series, lookback_m: int) -> pd.Series:
    return pressure_build.astype(int).rolling(lookback_m, min_periods=1).sum() >= 1


def compute_rv_from_close(close: pd.Series, window: int) -> pd.Series:
    # log returns
    r = np.log(close / close.shift(1))
    return r.rolling(window=window, min_periods=window).std()


def make_vol_trigger_cross(rv_fast: pd.Series, rv_slow: pd.Series) -> pd.Series:
    # Trigger when fast RV > slow RV (regime shift to higher vol)
    return rv_fast > rv_slow


def make_vol_trigger_drv(rv: pd.Series, lookback: int = 5, theta: float = 0.0) -> pd.Series:
    # Trigger when RV increases by more than theta over lookback days
    return (rv - rv.shift(lookback)) > theta


def backtest_fixed_holding(
    df: pd.DataFrame,
    entry_flag: pd.Series,
    holding_days: int = 20,
    fee_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df[["date", "close"]].copy()
    out["ret_1d"] = out["close"].pct_change().fillna(0.0)

    pos = np.zeros(len(out), dtype=int)
    trades = []
    i = 0
    fee = fee_bps / 10000.0

    while i < len(out):
        if bool(entry_flag.iloc[i]):
            entry_i = i
            exit_i = min(i + holding_days, len(out) - 1)

            # enter at close -> earn returns starting next day
            if entry_i + 1 <= exit_i:
                pos[entry_i + 1 : exit_i + 1] = 1

            entry_date = out["date"].iloc[entry_i]
            exit_date = out["date"].iloc[exit_i]
            entry_px = float(out["close"].iloc[entry_i])
            exit_px = float(out["close"].iloc[exit_i])

            gross_ret = (exit_px / entry_px) - 1.0
            net_ret = (1 - fee) * (1 + gross_ret) * (1 - fee) - 1.0

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "holding_days": int(exit_i - entry_i),
                    "gross_ret": float(gross_ret),
                    "net_ret": float(net_ret),
                }
            )

            i = exit_i + 1
        else:
            i += 1

    out["position"] = pos
    out["strategy_ret_1d"] = out["position"].astype(float) * out["ret_1d"]
    out["equity"] = (1.0 + out["strategy_ret_1d"]).cumprod()
    out["buy_hold_equity"] = (1.0 + out["ret_1d"]).cumprod()

    return out, pd.DataFrame(trades)


def perf_stats(equity: pd.Series, daily_ret: pd.Series) -> dict:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    total_ret = equity.iloc[-1] - 1.0
    vol = daily_ret.std()
    mean = daily_ret.mean()
    sharpe_like = (mean / vol) if vol > 1e-12 else np.nan

    return {
        "total_return": float(total_ret),
        "max_drawdown": float(max_dd),
        "mean_daily_ret": float(mean),
        "vol_daily_ret": float(vol),
        "sharpe_like": float(sharpe_like) if sharpe_like == sharpe_like else np.nan,
    }


def main(
    setup_lookback_m: int = 20,
    holding_days_h: int = 20,
    fee_bps: float = 0.0,
    # vol trigger config
    use_cross_trigger: bool = True,
    rv_fast_window: int = 5,
    rv_slow_window: int = 20,
    drv_lookback: int = 5,
    drv_theta: float = 0.0,
) -> None:
    root = project_root()
    states_path = root / "data" / "processed" / "rv50_states.parquet"
    if not states_path.exists():
        raise FileNotFoundError(f"States file not found: {states_path}")

    df = pd.read_parquet(states_path).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Setup
    setup = make_setup_flag(df["pressure_build"], lookback_m=setup_lookback_m)

    # Vol metrics (prefer using existing rv_20 if present; compute rv_5 anyway)
    if "rv_20" in df.columns:
        rv20 = df["rv_20"].astype(float)
    else:
        rv20 = compute_rv_from_close(df["close"], window=20)

    rv5 = compute_rv_from_close(df["close"], window=rv_fast_window)

    df["rv_fast"] = rv5
    df["rv_slow"] = rv20

    # Trigger
    if use_cross_trigger:
        trigger = make_vol_trigger_cross(df["rv_fast"], df["rv_slow"])
        trigger_name = f"rv{rv_fast_window}_gt_rv20"
    else:
        trigger = make_vol_trigger_drv(df["rv_slow"], lookback=drv_lookback, theta=drv_theta)
        trigger_name = f"drv{drv_lookback}_theta{drv_theta}"

    entry = setup & trigger

    curve, trades = backtest_fixed_holding(
        df=df,
        entry_flag=entry,
        holding_days=holding_days_h,
        fee_bps=fee_bps,
    )

    # Save outputs
    curve_path = root / "data" / "processed" / f"rv50_backtest_voltrigger_{trigger_name}.parquet"
    trades_path = root / "data" / "processed" / f"rv50_trades_voltrigger_{trigger_name}.parquet"
    curve.to_parquet(curve_path, index=False)
    trades.to_parquet(trades_path, index=False)

    print(f"Saved curve:  {curve_path}")
    print(f"Saved trades: {trades_path}")
    print(f"Trades: {len(trades)}")
    if len(trades) > 0:
        print(trades.tail(10))

    stats = perf_stats(curve["equity"], curve["strategy_ret_1d"])
    bh_stats = perf_stats(curve["buy_hold_equity"], curve["ret_1d"])

    print("\n=== Strategy stats (sample) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Buy&Hold stats (sample) ===")
    for k, v in bh_stats.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Signals count ===")
    print("pressure_build True:", int(df["pressure_build"].sum()))
    print("setup True:", int(setup.sum()))
    print("trigger True:", int(trigger.sum()))
    print("entry True:", int(entry.sum()))


if __name__ == "__main__":
    main(
        setup_lookback_m=20,
        holding_days_h=20,
        fee_bps=0.0,
        use_cross_trigger=True,   # 默认：rv5 > rv20
        rv_fast_window=5,
        rv_slow_window=20,
        drv_lookback=5,
        drv_theta=0.0,
    )
