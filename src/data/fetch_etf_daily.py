import pandas as pd
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def fetch_etf_daily(symbol: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date="20000101",
        end_date="20500101",
        adjust=""
    )

    df = raw.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    })

    keep = [
        c for c in
        ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
        if c in df.columns
    ]
    df = df[keep]

    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["date", "close"])
          .drop_duplicates(subset=["date"])
          .sort_values("date")
          .reset_index(drop=True)
    )

    return df


def save_etf_daily(df: pd.DataFrame, symbol: str) -> Path:
    root = project_root()
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{symbol}_daily.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = out_dir / f"{symbol}_daily.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")

    return path


if __name__ == "__main__":
    symbol = "588080"
    df = fetch_etf_daily(symbol)
    path = save_etf_daily(df, symbol)

    print("Saved to:", path)
    print("Rows:", len(df))
    print("Date range:", df["date"].min().date(), "->", df["date"].max().date())
    print(df.tail(5))
