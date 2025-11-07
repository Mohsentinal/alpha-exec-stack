from pathlib import Path
import os
import polars as pl
from loguru import logger

# -----------------------
# CONFIG (env-driven)
# -----------------------
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL   = os.getenv("SYMBOL",   "BTCUSDT")

RESAMPLE_MS = os.getenv("RESAMPLE_MS", "200ms").strip().lower()
if RESAMPLE_MS.isdigit():
    RESAMPLE_MS = f"{RESAMPLE_MS}ms"
EVERY       = RESAMPLE_MS
MS_INT      = int(RESAMPLE_MS.replace("ms", ""))
FWD_SECS    = int(os.getenv("FWD_SECS", "2"))
H_1S        = max(1, 1000 // MS_INT)
H_FWD       = max(1, (1000 // MS_INT) * FWD_SECS)

BASE = Path(__file__).resolve().parents[1]
TOB_FEAT = BASE / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*" / f"features_tob_resample-{EVERY}.parquet"
TRADES  = BASE / "data" / "trades"       / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

OUT = BASE / "data" / "features_tobtrades" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove(); logger.add(lambda m: print(m, end=""))


def load_tob():
    return pl.read_parquet(str(TOB_FEAT)).sort("ts")


def load_trades_lazy():
    return (
        pl.scan_parquet(str(TRADES / "date=*" / "hour=*" / "trades_*.parquet"))
          .select(
              pl.col("ts_trade_ms").cast(pl.Int64),
              pl.col("notional_usdt").cast(pl.Float64),
              pl.col("is_buyer_maker").cast(pl.Boolean),
          )
          .with_columns(ts = pl.from_epoch(pl.col("ts_trade_ms"), time_unit="ms"))
          .sort("ts")
    )


def bucket_trades(lf: pl.LazyFrame) -> pl.DataFrame:
    df = (
        lf.group_by_dynamic(index_column="ts", every=EVERY, period=EVERY, closed="right")
          .agg([
              # buyer is maker ⇒ seller is taker ⇒ trade hit the BID
              pl.col("notional_usdt").filter(pl.col("is_buyer_maker")).sum().alias("hit_bid_notional"),
              # buyer is taker ⇒ trade hit the ASK
              pl.col("notional_usdt").filter(~pl.col("is_buyer_maker")).sum().alias("hit_ask_notional"),
          ])
          .fill_null(0.0)
          .collect()
          .sort("ts")
    )

    # forward rolling sums we care about (1s, FWD_SECS, and a longer 5s for experiments)
    df = df.with_columns(
        hit_bid_notional_fwd1s = pl.col("hit_bid_notional").rolling_sum(window_size=H_1S, min_samples=1).shift(-(H_1S-1)),
        hit_ask_notional_fwd1s = pl.col("hit_ask_notional").rolling_sum(window_size=H_1S, min_samples=1).shift(-(H_1S-1)),
    )

    # dynamic horizon
    fwd_bid = pl.col("hit_bid_notional").rolling_sum(window_size=H_FWD, min_samples=1).shift(-(H_FWD-1))
    fwd_ask = pl.col("hit_ask_notional").rolling_sum(window_size=H_FWD, min_samples=1).shift(-(H_FWD-1))
    df = df.with_columns(
        fwd_bid.alias(f"hit_bid_notional_fwd{FWD_SECS}s"),
        fwd_ask.alias(f"hit_ask_notional_fwd{FWD_SECS}s"),
    )

    # optional 5s for analysis
    H_5S = max(1, 5 * (1000 // MS_INT))
    df = df.with_columns(
        pl.col("hit_bid_notional").rolling_sum(window_size=H_5S, min_samples=1).shift(-(H_5S-1)).alias("hit_bid_notional_fwd5s"),
        pl.col("hit_ask_notional").rolling_sum(window_size=H_5S, min_samples=1).shift(-(H_5S-1)).alias("hit_ask_notional_fwd5s"),
    )
    return df


def main():
    tob = load_tob()
    lf_tr = load_trades_lazy()
    tr = bucket_trades(lf_tr)

    # left-join trades onto our TOB grid
    df = tob.join(tr, on="ts", how="left").fill_null(0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        out_dir = OUT / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"features_tobtrades_resample-{EVERY}.parquet"
        df.filter(pl.col("date")==d).write_parquet(out)
        logger.info(f"→ wrote {out}\n")

    # quick look
    cols = [
        "ts", "spread_bps", "imb", "ofi" if "ofi" in df.columns else "microprice",
        "hit_bid_notional", "hit_ask_notional",
        "hit_bid_notional_fwd1s", "hit_ask_notional_fwd1s",
        f"hit_bid_notional_fwd{FWD_SECS}s", f"hit_ask_notional_fwd{FWD_SECS}s",
    ]
    print(df.select([c for c in cols if c in df.columns]).head(8))


if __name__ == "__main__":
    main()
