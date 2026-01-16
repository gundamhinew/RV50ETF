import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # RV50ETF
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# src/strategy/run_backtest.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# =========================
# Project root
# =========================
def project_root() -> Path:
    # .../RV50ETF/src/strategy/run_backtest.py -> parents[2] = RV50ETF
    return Path(__file__).resolve().parents[2]


# =========================
# Import pipeline modules (S1/S2/S3) + strategy core (S6)
# =========================
# S1: fetch ETF daily -> data/processed/{symbol}_daily.parquet
from src.data.fetch_etf_daily import fetch_etf_daily

# S2: build universe pressure -> data/processed/{universe}_pressure_main.parquet
from src.features.build_universe_pressure import build_universe_pressure, save_universe_pressure

# S3: build states -> data/processed/{symbol}_{universe}_states.parquet
from src.features.make_feature_table import make_feature_table, save_states

# S6: backtest core (this file is in the same folder)
from backtest_breakout import (
    make_setup_flag,
    make_vol_trigger_cross,
    make_vol_trigger_drv,
    backtest_fixed_holding,
    perf_stats,
)


# =========================
# Ensure artifacts exist
# =========================
def ensure_etf_daily(symbol: str, refresh: bool = False) -> Path:
    root = project_root()
    path = root / "data" / "processed" / f"{symbol}_daily.parquet"
    if path.exists() and not refresh:
        print(f"[S1] OK: {path}")
        return path

    print(f"[S1] Building ETF daily for {symbol} (refresh={refresh}) ...")
    fetch_etf_daily(symbol=symbol, refresh=refresh)

    if not path.exists():
        raise FileNotFoundError(f"[S1] ETF daily not generated: {path}")
    print(f"[S1] DONE: {path}")
    return path


def ensure_pressure(universe: str, refresh: bool = False, winsorize: bool = True) -> Path:
    root = project_root()
    path = root / "data" / "processed" / f"{universe}_pressure_main.parquet"
    if path.exists() and not refresh:
        print(f"[S2] OK: {path}")
        return path

    print(f"[S2] Building pressure for universe={universe} (refresh={refresh}, winsorize={winsorize}) ...")
    df = build_universe_pressure(universe=universe, refresh=refresh, winsorize=winsorize)
    out_path = save_universe_pressure(df, universe=universe)

    if not out_path.exists():
        raise FileNotFoundError(f"[S2] Pressure not generated: {out_path}")
    print(f"[S2] DONE: {out_path}")
    return out_path


def ensure_states(symbol: str, universe: str, refresh: bool = False) -> Path:
    root = project_root()
    path = root / "data" / "processed" / f"{symbol}_{universe}_states.parquet"
    if path.exists() and not refresh:
        print(f"[S3] OK: {path}")
        return path

    print(f"[S3] Building states for symbol={symbol}, universe={universe} (refresh={refresh}) ...")
    df = make_feature_table(symbol=symbol, universe=universe, refresh=refresh)
    out_path = save_states(df, symbol=symbol, universe=universe)

    if not out_path.exists():
        raise FileNotFoundError(f"[S3] States not generated: {out_path}")
    print(f"[S3] DONE: {out_path}")
    return out_path


# =========================
# Run backtest
# =========================
def run_backtest(
    symbol: str,
    universe: str,
    holding_days: int = 20,
    use_cross_trigger: bool = True,
    fee_bps: float = 0.0,
    refresh: bool = False,
    winsorize: bool = True,
) -> None:
    root = project_root()
    print(f"\n=== RUN BACKTEST ===")
    print(f"symbol={symbol}  universe={universe}")
    print(f"holding_days={holding_days}  use_cross_trigger={use_cross_trigger}  fee_bps={fee_bps}")
    print(f"refresh={refresh}  winsorize={winsorize}\n")

    # 1) Ensure upstream artifacts
    ensure_etf_daily(symbol, refresh=refresh)
    ensure_pressure(universe, refresh=refresh, winsorize=winsorize)
    states_path = ensure_states(symbol, universe, refresh=refresh)

    # 2) Load states
    df = pd.read_parquet(states_path)
    if "date" not in df.columns:
        raise KeyError(f"[S3] states missing 'date': columns={df.columns.tolist()}")
    if "close" not in df.columns:
        raise KeyError(f"[S3] states missing 'close': columns={df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Required signals from S3
    for col in ["pressure_build", "rv_5", "rv_20"]:
        if col not in df.columns:
            raise KeyError(f"[S3] states missing '{col}': columns={df.columns.tolist()}")

    # 3) Setup: convert sparse pressure_build into a tradable 'setup' window
    setup = make_setup_flag(df["pressure_build"], lookback_m=20)

    # 4) Trigger: "升波触发"
    if use_cross_trigger:
        trigger = make_vol_trigger_cross(df["rv_5"], df["rv_20"])
    else:
        trigger = make_vol_trigger_drv(df["rv_20"], lookback=5, theta=0.01)

    entry = setup & trigger

    # 5) Backtest
    curve, trades = backtest_fixed_holding(
        price_df=df[["date", "close"]].copy(),
        entry_flag=entry,
        holding_days=holding_days,
        fee_bps=fee_bps,
    )

    # 6) Save outputs
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / f"{symbol}_{universe}_backtest_curve.parquet"
    trades_path = out_dir / f"{symbol}_{universe}_backtest_trades.parquet"
    curve.to_parquet(curve_path, index=False)
    trades.to_parquet(trades_path, index=False)

    # 7) Print stats
    strat = perf_stats(curve["equity"], curve["strategy_ret_1d"])
    bh = perf_stats(curve["buy_hold_equity"], curve["ret_1d"])

    print(f"\nStates: {states_path}")
    print(f"Saved curve: {curve_path}")
    print(f"Saved trades: {trades_path}")

    print("\n=== Strategy stats ===")
    for k, v in strat.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Buy&Hold stats ===")
    for k, v in bh.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Signals count ===")
    print(f"pressure_build True: {int(df['pressure_build'].sum())}")
    print(f"setup True: {int(setup.sum())}")
    print(f"trigger True: {int(trigger.sum())}")
    print(f"entry True: {int(entry.sum())}")
    print(f"Trades: {len(trades)}")
    print("\n=== DONE ===\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command pipeline: S1->S2->S3->S6")
    parser.add_argument("--symbol", type=str, required=True, help="ETF symbol, e.g. 588080 / 159919")
    parser.add_argument("--universe", type=str, required=True, help="universe name, e.g. kc50 / hs300")
    parser.add_argument("--holding_days", type=int, default=20)
    parser.add_argument("--use_cross_trigger", type=str, default="true")
    parser.add_argument("--fee_bps", type=float, default=0.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no_winsorize", action="store_true")

    args = parser.parse_args()
    use_cross = args.use_cross_trigger.lower() in ("1", "true", "yes", "y")

    run_backtest(
        symbol=args.symbol,
        universe=args.universe,
        holding_days=args.holding_days,
        use_cross_trigger=use_cross,
        fee_bps=args.fee_bps,
        refresh=args.refresh,
        winsorize=(not args.no_winsorize),
    )


if __name__ == "__main__":
    main()
