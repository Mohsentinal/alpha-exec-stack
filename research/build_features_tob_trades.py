from pathlib import Path
import os
import polars as pl
from loguru import logger

# =========================
# CONFIG (env-overridable)
# =========================
EXCHANGE = "binance"
SYMBOL   = "BTCUSDT"

RESAMPLE_MS = int(os.getenv("RESAMPLE_MS", "200"))  # 200 or 100, etc.
EVERY       = f"{RESAMPLE_MS}ms"
STEPS_PER_SEC = max(1, 1000 // RESAMPLE_MS)

# Forward windows (seconds) to pre-compute
FWD_LIST = [1, 2, 5]

BASE = Path(__file__).resolve().parents[1]
TOB_FEAT = BASE / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*" / f"features_tob_resample-{EVERY}.parquet"
TRADES   = BASE / "data" / "trades"      / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"
OUT      = BASE / "data" / "features_tobtrades" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_tob() -> pl.DataFrame:
    return pl.read_parquet(str(TOB_FEAT)).sort("ts")


def load_trades_lazy() -> pl.LazyFrame:
    return (
        pl.scan_parquet(str(TRADES / "date=*" / "hour=*" / "trades_*.parquet"))
        .select(
            pl.col("ts_trade_ms").cast(pl.Int64),
            pl.col("notional_usdt").cast(pl.Float64),
            pl.col("is_buyer_maker").cast(pl.Boolean),
        )
        .with_columns(ts=pl.from_epoch(pl.col("ts_trade_ms"), time_unit="ms"))
        .sort("ts")
    )


def bucket_trades(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Aggregate trades onto the same time grid as TOB features.
    Also compute forward rolling sums of aggressive notional for multiple horizons.
    """
    df = (
        lf.group_by_dynamic(index_column="ts", every=EVERY, period=EVERY, closed="right")
        .agg([
            # Binance: is_buyer_maker=True -> buyer is maker, seller is taker -> trade hits BID
            pl.col("notional_usdt").filter(pl.col("is_buyer_maker")).sum().alias("hit_bid_notional"),
            # Otherwise trade hits ASK
            pl.col("notional_usdt").filter(~pl.col("is_buyer_maker")).sum().alias("hit_ask_notional"),
        ])
        .fill_null(0.0)
        .collect()
        .sort("ts")
    )

    # Add forward sums for each requested horizon
    with_cols = []
    for secs in FWD_LIST:
        steps = max(1, (secs * 1000) // RESAMPLE_MS)
        with_cols += [
            pl.col("hit_bid_notional").rolling_sum(window_size=steps, min_samples=1).shift(-(steps - 1)).alias(f"hit_bid_notional_fwd{secs}s"),
            pl.col("hit_ask_notional").rolling_sum(window_size=steps, min_samples=1).shift(-(steps - 1)).alias(f"hit_ask_notional_fwd{secs}s"),
        ]
    return df.with_columns(with_cols)


def main():
    tob = load_tob()
    lf_tr = load_trades_lazy()
    tr = bucket_trades(lf_tr)

    # Left-join trades onto TOB grid
    df = tob.join(tr, on="ts", how="left").fill_null(0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        out_dir = OUT / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"features_tobtrades_resample-{EVERY}.parquet"
        df.filter(pl.col("date") == d).write_parquet(out)
        logger.info(f"→ wrote {out}\n")

    # Quick peek
    head_cols = [
        "ts", "spread_bps", "imb", "ofi", "microprice",
        "hit_bid_notional", "hit_ask_notional",
        "hit_bid_notional_fwd1s", "hit_ask_notional_fwd1s",
        "hit_bid_notional_fwd2s", "hit_ask_notional_fwd2s",
        "hit_bid_notional_fwd5s", "hit_ask_notional_fwd5s",
    ]
    print(df.select([c for c in head_cols if c in df.columns]).head(8))


if __name__ == "__main__":
    main()
