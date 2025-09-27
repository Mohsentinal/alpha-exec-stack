# research/build_features.py
# Build top-of-book features (best bid/ask), midprice, spread, and short-horizon markouts
# from the Parquet files created by ingest/binance_l2_ingest.py.

from pathlib import Path
import sys
import polars as pl
from loguru import logger

# ----------------------------
# Config (edit these safely)
# ----------------------------
EXCHANGE = "binance"
SYMBOL = "BTCUSDT"        # folders are written in UPPERCASE by the ingestor
RESAMPLE_EVERY = "1s"     # e.g. "100ms", "500ms", "1s"
H_1S, H_5S, H_30S = 1, 5, 30   # horizons in resampled steps

# ----------------------------
# Paths (robust to any working directory)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # .../alpha-exec-stack

# Primary (correct) location where the ingestor now writes:
PRIMARY_ROOT = BASE_DIR / "data" / "quotes" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

# Fallback for any older runs that may have landed under ingest/data:
FALLBACK_ROOT = BASE_DIR / "ingest" / "data" / "quotes" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

FEATURES_ROOT = BASE_DIR / "data" / "features" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove()
logger.add(lambda m: print(m, end=""), level="INFO")


def rglob_files(root: Path) -> list[str]:
    """Recursively collect parquet paths under root."""
    return sorted(str(p) for p in root.rglob("l2_*.parquet"))


def find_input_files() -> list[str]:
    """Look for input parquet files in primary location, then fallback."""
    files = rglob_files(PRIMARY_ROOT)
    if files:
        logger.info(f"Using quotes under:\n  {PRIMARY_ROOT}\nFound {len(files)} files.\n")
        return files

    fb = rglob_files(FALLBACK_ROOT)
    if fb:
        logger.info(f"(Fallback) Using quotes under:\n  {FALLBACK_ROOT}\nFound {len(fb)} files.\n")
        return fb

    logger.error(
        "No parquet files found in either location:\n"
        f"  {PRIMARY_ROOT}\n"
        f"  {FALLBACK_ROOT}\n"
        "Run the ingestor for ~1–2 minutes, then retry."
    )
    sys.exit(1)


def load_l2_updates_lazy(files: list[str]) -> pl.LazyFrame:
    """Lazily scan parquet files, keep required columns with stable types."""
    return (
        pl.scan_parquet(files)
        .select(
            pl.col("ts_event_ms").cast(pl.Int64),
            pl.col("side").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
        )
    )


def compute_best_quotes(lf_updates: pl.LazyFrame) -> pl.LazyFrame:
    """
    For each event time, compute best bid (max bid) and best ask (min ask),
    then forward-fill to ensure both sides exist across time.
    """
    return (
        lf_updates
        .group_by("ts_event_ms")
        .agg([
            pl.col("price").filter(pl.col("side") == "bid").max().alias("best_bid"),
            pl.col("price").filter(pl.col("side") == "ask").min().alias("best_ask"),
        ])
        .with_columns(ts=pl.from_epoch(pl.col("ts_event_ms"), time_unit="ms"))
        .sort("ts")
        .with_columns(
            pl.col("best_bid").forward_fill(),
            pl.col("best_ask").forward_fill(),
        )
        .select("ts_event_ms", "ts", "best_bid", "best_ask")
    )


def resample_top_of_book(lf_best: pl.LazyFrame, every: str) -> pl.LazyFrame:
    """
    Resample to a regular grid (e.g., 1s), taking the last known best bid/ask in each bucket.
    """
    return (
        lf_best
        .group_by_dynamic(index_column="ts", every=every, period=every, closed="right")
        .agg([
            pl.col("ts_event_ms").last().alias("ts_event_ms"),
            pl.col("best_bid").last().alias("best_bid"),
            pl.col("best_ask").last().alias("best_ask"),
        ])
        .drop_nulls(["best_bid", "best_ask"])
        .with_columns(
            mid=((pl.col("best_bid") + pl.col("best_ask")) / 2.0),
            spread=(pl.col("best_ask") - pl.col("best_bid")),
        )
        .with_columns(
            spread_bps=(pl.col("spread") / pl.col("mid") * 1e4),
            date=pl.col("ts").dt.strftime("%Y-%m-%d"),
            hour=pl.col("ts").dt.strftime("%H"),
        )
    )


def add_forward_markouts(lf: pl.LazyFrame, h1: int, h5: int, h30: int) -> pl.LazyFrame:
    """Add forward midprice and returns (markouts) and simple direction labels."""
    return (
        lf
        .with_columns(
            mid_fwd_1=pl.col("mid").shift(-h1),
            mid_fwd_5=pl.col("mid").shift(-h5),
            mid_fwd_30=pl.col("mid").shift(-h30),
        )
        .with_columns(
            ret_1s=(pl.col("mid_fwd_1") - pl.col("mid")) / pl.col("mid"),
            ret_5s=(pl.col("mid_fwd_5") - pl.col("mid")) / pl.col("mid"),
            ret_30s=(pl.col("mid_fwd_30") - pl.col("mid")) / pl.col("mid"),
        )
        .with_columns(
            dir_1s=pl.when(pl.col("ret_1s") > 0).then(1).when(pl.col("ret_1s") < 0).then(-1).otherwise(0),
            dir_5s=pl.when(pl.col("ret_5s") > 0).then(1).when(pl.col("ret_5s") < 0).then(-1).otherwise(0),
            dir_30s=pl.when(pl.col("ret_30s") > 0).then(1).when(pl.col("ret_30s") < 0).then(-1).otherwise(0),
        )
    )


def write_features(df: pl.DataFrame):
    """Write features partitioned by date."""
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        out_dir = FEATURES_ROOT / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"features_resample-{RESAMPLE_EVERY}.parquet"
        df.filter(pl.col("date") == d).write_parquet(out_path)
        logger.info(f"→ wrote {out_path}\n")


def main():
    files = find_input_files()

    logger.info("Loading L2 updates lazily…\n")
    lf_updates = load_l2_updates_lazy(files)

    logger.info("Computing per-timestamp best bid/ask with forward-fill…\n")
    lf_best = compute_best_quotes(lf_updates)

    logger.info(f"Resampling top-of-book every {RESAMPLE_EVERY}…\n")
    lf_resampled = resample_top_of_book(lf_best, RESAMPLE_EVERY)

    logger.info("Adding forward markouts (1s, 5s, 30s)…\n")
    lf_features = add_forward_markouts(lf_resampled, H_1S, H_5S, H_30S)

    logger.info("Collecting and writing features…\n")
    df = lf_features.collect()
    write_features(df)

    # Safe preview (no .to_string() dependency)
    sample = df.select("ts", "mid", "spread_bps", "ret_1s", "ret_5s", "ret_30s").head(8)
    logger.info(f"\n{sample}\n")


if __name__ == "__main__":
    main()
