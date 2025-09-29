# research/build_features_tob.py
from pathlib import Path
import polars as pl
from loguru import logger
import os

# ---- Config ----
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL   = os.getenv("SYMBOL", "BTCUSDT")
EVERY    = os.getenv("RESAMPLE_MS", "200ms")            # resample grid
H_1S, H_5S, H_30S = 5, 25, 150   # horizons in resampled steps for 100ms grid
EPS = 1e-12

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parents[1]
TOB_ROOT = BASE_DIR / "data" / "tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"
OUT_ROOT = BASE_DIR / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove()
logger.add(lambda m: print(m, end=""))

def find_files() -> list[str]:
    files = sorted(str(p) for p in TOB_ROOT.rglob("tob_*.parquet"))
    logger.info(f"Searching TOB under:\n  {TOB_ROOT}\nFound {len(files)} files.\n")
    return files

def load_tob_lazy(files: list[str]) -> pl.LazyFrame:
    # Keep only raw fields; add 'ts' as datetime for grouping
    return (
        pl.scan_parquet(files)
        .select(
            pl.col("ts_recv_ms").cast(pl.Int64),
            pl.col("bid_px").cast(pl.Float64),
            pl.col("bid_sz").cast(pl.Float64),
            pl.col("ask_px").cast(pl.Float64),
            pl.col("ask_sz").cast(pl.Float64),
        )
        .with_columns(ts=pl.from_epoch(pl.col("ts_recv_ms"), time_unit="ms"))
        .sort("ts")
    )

def resample_raw_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Only resample & take last raw values per bucket (no derived cols here)
    return (
        lf.group_by_dynamic(index_column="ts", every=EVERY, period=EVERY, closed="right")
        .agg([
            pl.col("ts_recv_ms").last().alias("ts_recv_ms"),
            pl.col("bid_px").last().alias("bid_px"),
            pl.col("ask_px").last().alias("ask_px"),
            pl.col("bid_sz").last().alias("bid_sz"),
            pl.col("ask_sz").last().alias("ask_sz"),
        ])
        .drop_nulls(["bid_px", "ask_px", "bid_sz", "ask_sz"])
    )

def add_derived_eager(df: pl.DataFrame) -> pl.DataFrame:
    # Compute all derived columns in eager mode (prevents lazy pruning issues)
    df = df.sort("ts").with_columns(
        mid=(pl.col("bid_px") + pl.col("ask_px")) / 2.0,
        spread=pl.col("ask_px") - pl.col("bid_px"),
    ).with_columns(
        spread_bps=(pl.col("spread") / pl.col("mid") * 1e4),
        imb=(pl.col("bid_sz") - pl.col("ask_sz")) / (pl.col("bid_sz") + pl.col("ask_sz") + EPS),
        microprice=(
            pl.col("ask_px") * pl.col("bid_sz") + pl.col("bid_px") * pl.col("ask_sz")
        ) / (pl.col("bid_sz") + pl.col("ask_sz") + EPS),
        date=pl.col("ts").dt.strftime("%Y-%m-%d"),
        hour=pl.col("ts").dt.strftime("%H"),
    )
    return df

def add_ofi_eager(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort("ts")
        .with_columns(
            bid_px_prev=pl.col("bid_px").shift(1),
            ask_px_prev=pl.col("ask_px").shift(1),
            bid_sz_prev=pl.col("bid_sz").shift(1),
            ask_sz_prev=pl.col("ask_sz").shift(1),
        )
        .with_columns(
            ofi=(
                (pl.when(pl.col("bid_px") > pl.col("bid_px_prev")).then(pl.col("bid_sz")).otherwise(0.0))
                - (pl.when(pl.col("bid_px") < pl.col("bid_px_prev")).then(pl.col("bid_sz_prev")).otherwise(0.0))
                - (pl.when(pl.col("ask_px") < pl.col("ask_px_prev")).then(pl.col("ask_sz")).otherwise(0.0))
                + (pl.when(pl.col("ask_px") > pl.col("ask_px_prev")).then(pl.col("ask_sz_prev")).otherwise(0.0))
            )
        )
        .drop(["bid_px_prev", "ask_px_prev", "bid_sz_prev", "ask_sz_prev"])
    )

def add_markouts_eager(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort("ts")
        .with_columns(
            mid_fwd_1=pl.col("mid").shift(-H_1S),
            mid_fwd_5=pl.col("mid").shift(-H_5S),
            mid_fwd_30=pl.col("mid").shift(-H_30S),
        )
        .with_columns(
            ret_1s=(pl.col("mid_fwd_1") - pl.col("mid")) / pl.col("mid"),
            ret_5s=(pl.col("mid_fwd_5") - pl.col("mid")) / pl.col("mid"),
            ret_30s=(pl.col("mid_fwd_30") - pl.col("mid")) / pl.col("mid"),
        )
        .with_columns(
            dir_1s=pl.when(pl.col("ret_1s") > 0).then(1).when(pl.col("ret_1s") < 0).then(-1).otherwise(0),
        )
    )

def write_features(df: pl.DataFrame):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        out_dir = OUT_ROOT / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"features_tob_resample-{EVERY}.parquet"
        df.filter(pl.col("date") == d).write_parquet(out_path)
        logger.info(f"→ wrote {out_path}\n")

def main():
    files = find_files()
    if not files:
        logger.error("No TOB parquet found. Run binance_bookticker_ingest.py first for a few minutes.")
        return

    lf = load_tob_lazy(files)
    lf_resampled = resample_raw_lazy(lf)

    # MATERIALIZE the resampled raw TOB
    df = lf_resampled.collect()

    # Eager feature engineering
    df = add_derived_eager(df)
    df = add_ofi_eager(df)
    df = add_markouts_eager(df)

    write_features(df)

    sample = df.select("ts", "mid", "spread_bps", "imb", "ofi", "microprice", "ret_1s").head(10)
    logger.info(f"\n{sample}\n")

if __name__ == "__main__":
    main()
