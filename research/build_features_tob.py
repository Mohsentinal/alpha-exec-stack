from pathlib import Path
import os
import polars as pl
from loguru import logger

# -----------------------
# CONFIG (env-driven)
# -----------------------
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL   = os.getenv("SYMBOL",   "BTCUSDT")

# Resample step as a string Polars understands ("200ms", "100ms", etc.)
RESAMPLE_MS = os.getenv("RESAMPLE_MS", "200ms").strip().lower()
if RESAMPLE_MS.isdigit():
    RESAMPLE_MS = f"{RESAMPLE_MS}ms"  # guard against plain "200" (needs unit)
EVERY = RESAMPLE_MS

# Forward horizon (seconds)
FWD_SECS  = int(os.getenv("FWD_SECS", "2"))
MS_INT    = int(RESAMPLE_MS.replace("ms", ""))
H_FWD     = max(1, (1000 // MS_INT) * FWD_SECS)  # number of resampled steps in FWD_SECS

BASE = Path(__file__).resolve().parents[1]
RAW_TOB_GLOB = str(
    BASE / "data" / "tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*" / "hour=*" / "tob_*.parquet"
)
OUT_DIR = BASE / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove(); logger.add(lambda m: print(m, end=""))


def _pick_time_column(schema_names_lower: dict) -> tuple[str, str] | None:
    """
    Return (original_col_name, unit_hint) where unit_hint is one of {"ms","ns","auto"}.
    We search for many common timestamp field names.
    """
    candidates = [
        ("ts", "auto"),
        ("ts_ms", "ms"),
        ("ts_recv_ms", "ms"),        # ← your ingestor
        ("timestamp_ms", "ms"),
        ("event_time_ms", "ms"),
        ("event_ms", "ms"),
        ("e", "ms"),                 # Binance event time
        ("t", "ms"),
        ("time_ms", "ms"),
        ("event_time", "ms"),
        ("timestamp", "ms"),
        ("time", "ms"),
        ("recv_ts_ms", "ms"),
        ("ws_ts_ms", "ms"),
        ("ts_ns", "ns"),
        ("timestamp_ns", "ns"),
    ]
    for low, unit in candidates:
        if low in schema_names_lower:
            return schema_names_lower[low], unit
    return None


def _ensure_ts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ensure we have a datetime 'ts' column.
    Accepts many possible input names: ts, ts_ms, event_time_ms, ts_recv_ms, E, T, timestamp, etc.
    Converts ints to epoch ms (or ns), and parses strings when needed.
    """
    schema = lf.collect_schema()
    names = list(schema.keys())
    names_lower = {n.lower(): n for n in names}

    picked = _pick_time_column(names_lower)
    if not picked:
        raise KeyError(
            "Expected a time column (e.g., 'ts', 'ts_ms', 'ts_recv_ms', 'event_time_ms', 'E', 'T', "
            f"'timestamp_ms', etc.). Found columns: {names}"
        )

    src_name, unit_hint = picked
    src_dtype = schema[src_name]

    if src_dtype == pl.Datetime:
        ts_expr = pl.col(src_name).alias("ts")
    elif src_dtype == pl.Utf8:
        ts_expr = pl.col(src_name).str.strptime(pl.Datetime, strict=False).alias("ts")
    else:
        time_unit = "ns" if unit_hint == "ns" else "ms"
        ts_expr = pl.from_epoch(pl.col(src_name).cast(pl.Int64), time_unit=time_unit).alias("ts")

    return lf.with_columns(ts_expr)


def load_raw_lazy() -> pl.LazyFrame:
    """
    Load raw top-of-book snapshots written by the ingestor.
    We alias common variants to: bid, ask, bid_qty, ask_qty.
    """
    lf = pl.scan_parquet(RAW_TOB_GLOB)
    lf = _ensure_ts(lf)

    schema = lf.collect_schema()
    cols = {c.lower(): c for c in schema.keys()}

    def pick(*cands, default=None):
        for c in cands:
            if c in cols:
                return cols[c]
        return default

    # Your column names included here:
    bid_col = pick("bid", "best_bid", "b", "bid_px")
    ask_col = pick("ask", "best_ask", "a", "ask_px")
    bq_col  = pick("bid_qty", "bq", "bid_size", "best_bid_qty", "bidsz", "bid_sz")
    aq_col  = pick("ask_qty", "aq", "ask_size", "best_ask_qty", "asksz", "ask_sz")

    if bid_col is None or ask_col is None:
        raise KeyError(f"Could not find bid/ask columns in raw TOB files. Found: {list(schema.keys())}")

    selects = [
        pl.col("ts"),
        pl.col(bid_col).cast(pl.Float64).alias("bid"),
        pl.col(ask_col).cast(pl.Float64).alias("ask"),
    ]
    if bq_col:
        selects.append(pl.col(bq_col).cast(pl.Float64).alias("bid_qty"))
    else:
        selects.append(pl.lit(0.0).cast(pl.Float64).alias("bid_qty"))
    if aq_col:
        selects.append(pl.col(aq_col).cast(pl.Float64).alias("ask_qty"))
    else:
        selects.append(pl.lit(0.0).cast(pl.Float64).alias("ask_qty"))

    return lf.select(selects)


def resample_raw_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Resample to a fixed grid (EVERY), taking last values in each bucket.
    Then derive features on that SAME grid.

    IMPORTANT: don't reference 'mid' inside the same with_columns where it's created.
    """
    agg = (
        lf.group_by_dynamic(index_column="ts", every=EVERY, period=EVERY, closed="right")
          .agg([
              pl.col("bid").last().alias("bid"),
              pl.col("ask").last().alias("ask"),
              pl.col("bid_qty").last().alias("bid_qty"),
              pl.col("ask_qty").last().alias("ask_qty"),
          ])
    )

    # First, create 'mid'
    agg = agg.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid")
    )

    # Then compute the other features (avoid referencing 'mid' before it exists)
    agg = agg.with_columns([
        # spread in bps of mid — computed without referencing 'mid' in the same call it was created
        (((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2.0)) * 1e4).alias("spread_bps"),
        # imbalance
        ((pl.col("bid_qty") - pl.col("ask_qty")) / (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9)).alias("imb"),
        # microprice
        (((pl.col("ask") * pl.col("bid_qty")) + (pl.col("bid") * pl.col("ask_qty")))
         / (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9)).alias("microprice"),
    ])
    return agg


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add forward returns/labels for the configured horizon (FWD_SECS).
    Names are ret_{FWD_SECS}s and dir_{FWD_SECS}s.
    Also keep 1s legacy labels for compatibility.
    """
    ret_col = f"ret_{FWD_SECS}s"
    dir_col = f"dir_{FWD_SECS}s"

    df = df.with_columns(
        (pl.col("mid").shift(-H_FWD) / pl.col("mid") - 1.0).alias(ret_col)
    ).with_columns(
        pl.when(pl.col(ret_col) > 0).then(pl.lit(1))
         .when(pl.col(ret_col) < 0).then(pl.lit(-1))
         .otherwise(pl.lit(0)).alias(dir_col)
    )

    steps_1s = max(1, 1000 // MS_INT)
    df = df.with_columns(
        (pl.col("mid").shift(-steps_1s) / pl.col("mid") - 1.0).alias("ret_1s")
    ).with_columns(
        pl.when(pl.col("ret_1s") > 0).then(pl.lit(1))
         .when(pl.col("ret_1s") < 0).then(pl.lit(-1))
         .otherwise(pl.lit(0)).alias("dir_1s")
    )
    return df


def add_date(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("ts").dt.strftime("%Y-%m-%d").alias("date"))


def write_by_date(df: pl.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        out_dir = OUT_DIR / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"features_tob_resample-{EVERY}.parquet"
        df.filter(pl.col("date") == d).write_parquet(out)
        logger.info(f"→ wrote {out}\n")


def main():
    logger.info(f"[build_features_tob] using EVERY='{EVERY}' (MS={MS_INT}) | FWD_SECS={FWD_SECS} | H_FWD={H_FWD}\n")
    logger.info("Searching TOB under:\n  " + str(Path(RAW_TOB_GLOB).parent.parent.parent))

    lf_raw = load_raw_lazy()
    lf_resampled = resample_raw_lazy(lf_raw)
    df = lf_resampled.collect()

    df = add_returns(df)
    df = add_date(df)
    write_by_date(df)

    # quick head
    head_cols = ["ts", "mid", "spread_bps", "imb", "microprice", f"ret_{FWD_SECS}s"]
    print(df.select(head_cols).head(10))


if __name__ == "__main__":
    main()
