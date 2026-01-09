from __future__ import annotations

from pathlib import Path
import pandas as pd


# =========================
# Project root helper
# =========================
def project_root() -> Path:
    # build_kc50_pressure.py is under RV50ETF/src/features/
    # parents[0]=features, parents[1]=src, parents[2]=RV50ETF
    return Path(__file__).resolve().parents[2]


# =========================
# 1) Constituents
# =========================
def get_kc50_constituents() -> list[str]:
    """
    Load STAR50 constituents from local static file:
      RV50ETF/data/raw/kc50_constituents.csv
    CSV format:
      stock_code
      688xxx
      ...
    """
    root = project_root()
    path = root / "data" / "raw" / "kc50_constituents.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing constituents file: {path}")

    df = pd.read_csv(path, dtype={"stock_code": str})
    if "stock_code" not in df.columns:
        raise ValueError(f"kc50_constituents.csv must contain column 'stock_code'. Got: {df.columns.tolist()}")

    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.zfill(6)

    # drop empty rows / duplicates
    df = df[df["stock_code"].str.match(r"^\d{6}$", na=False)].drop_duplicates(subset=["stock_code"])
    return df["stock_code"].tolist()


# =========================
# 2) Fetch + cache fund flow for each stock
# =========================
def fetch_stock_main_flow(stock_code: str) -> pd.DataFrame:
    """
    Fetch individual stock fund flow (Eastmoney via AkShare), standardized schema:
      date, stock_code, main_net_inflow

    NOTE: This is "成交归因型" net flow proxy, NOT true position change.
    """
    import akshare as ak

    market = "sh" if stock_code.startswith("6") else "sz"
    df = ak.stock_individual_fund_flow(stock=stock_code, market=market)

    # Date column
    if "日期" in df.columns:
        date_col = "日期"
    elif "date" in df.columns:
        date_col = "date"
    else:
        raise ValueError(f"[{stock_code}] no date column in: {df.columns.tolist()}")

    # Main net inflow column (most common naming)
    candidates = ["主力净流入-净额", "主力净流入净额", "主力净流入", "主力净额"]
    main_col = next((c for c in candidates if c in df.columns), None)
    if main_col is None:
        raise ValueError(f"[{stock_code}] no main inflow column in: {df.columns.tolist()}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "stock_code": stock_code,
        "main_net_inflow": pd.to_numeric(df[main_col], errors="coerce"),
    })

    out = (
        out.dropna(subset=["date"])
           .drop_duplicates(subset=["date"])
           .sort_values("date")
           .reset_index(drop=True)
    )
    return out


def load_or_fetch_stock_flow(stock_code: str, refresh: bool = False) -> pd.DataFrame:
    """
    Cache each stock's fund flow to:
      RV50ETF/data/raw/fund_flow/{stock_code}.parquet
    """
    root = project_root()
    raw_dir = root / "data" / "raw" / "fund_flow"
    raw_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{stock_code}.parquet"

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = fetch_stock_main_flow(stock_code)
    df.to_parquet(path, index=False)
    return df


# =========================
# 3) Aggregate to index-level pressure
# =========================
def winsorize_by_stock(df: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """
    Winsorize inflow within each stock to reduce outliers.
    """
    def _clip(s: pd.Series) -> pd.Series:
        lo = s.quantile(p)
        hi = s.quantile(1 - p)
        return s.clip(lo, hi)

    out = df.copy()
    out["main_net_inflow"] = out.groupby("stock_code")["main_net_inflow"].transform(_clip)
    return out


def build_kc50_pressure(refresh: bool = False, winsorize: bool = True) -> pd.DataFrame:
    """
    Build index-level 'pressure' by summing constituent main_net_inflow per day.

    Output columns:
      date
      pressure_main
      pressure_main_20 (20d rolling sum)
      pressure_main_z60 (60d zscore)
    """
    codes = get_kc50_constituents()
    print(f"Constituents: {len(codes)}")

    frames: list[pd.DataFrame] = []
    for code in codes:
        try:
            frames.append(load_or_fetch_stock_flow(code, refresh=refresh))
        except Exception as e:
            print(f"[WARN] {code} fetch failed: {e}")

    if not frames:
        raise RuntimeError("No fund flow data fetched. Check AkShare availability / network.")

    all_flow = pd.concat(frames, ignore_index=True)

    if winsorize:
        all_flow = winsorize_by_stock(all_flow, p=0.01)

    # Equal-weight aggregation: sum of main_net_inflow across constituents per day
    agg = (
        all_flow.groupby("date", as_index=False)["main_net_inflow"]
                .sum()
                .rename(columns={"main_net_inflow": "pressure_main"})
                .sort_values("date")
                .reset_index(drop=True)
    )

    # Rolling features (optional but useful)
    agg["pressure_main_20"] = agg["pressure_main"].rolling(20, min_periods=10).sum()
    agg["pressure_main_z60"] = (
        (agg["pressure_main"] - agg["pressure_main"].rolling(60, min_periods=30).mean())
        / agg["pressure_main"].rolling(60, min_periods=30).std()
    )

    return agg


def main() -> None:
    root = project_root()
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    out = build_kc50_pressure(refresh=False, winsorize=True)

    out_path = proc_dir / "kc50_pressure_main.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out)}  Date range: {out['date'].min().date()} -> {out['date'].max().date()}")
    print(out.tail(5))


if __name__ == "__main__":
    main()
