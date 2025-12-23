from __future__ import annotations

from pathlib import Path
import os
import re
import glob
import polars as pl
from loguru import logger

# -----------------------
# CONFIG (env-driven)
# -----------------------
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

# Optional: choose which date to process
# - DATE not set  -> auto-pick newest date=YYYY-MM-DD folder
# - DATE=2099-01-01 -> only that date
# - DATE=*        -> all dates (requires consistent parquet schema across all files)
DATE = os.getenv("DATE", "").strip()

# Resample step as a string Polars understands ("200ms", "100ms", etc.)
RESAMPLE_MS = os.getenv("RESAMPLE_MS", "200ms").strip().lower()
if RESAMPLE_MS.isdigit():
    RESAMPLE_MS = f"{RESAMPLE_MS}ms"
EVERY = RESAMPLE_MS

# Forward horizon (seconds)
FWD_SECS = int(os.getenv("FWD_SECS", "2"))
MS_INT = int(RESAMPLE_MS.replace("ms", ""))
H_FWD = max(1, (1000 // MS_INT) * FWD_SECS)

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "data" / "tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

OUT_DIR = BASE / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove()
logger.add(lambda m: print(m, end=""))


def _list_dates(root: Path) -> list[str]:
    """Return list of available dates like ['2025-12-17', '2099-01-01', ...]."""
    if not root.exists():
        return []
    dates = []
    for p in root.glob("date=*"):
        if p.is_dir():
            m = re.match(r"date=(\d{4}-\d{2}-\d{2})$", p.name)
            if m:
                dates.append(m.group(1))
    return sorted(set(dates))


def _pick_default_date(root: Path) -> str | None:
    dates = _list_dates(root)
    return max(dates) if dates else None


def _scan_parquet_compat(files: list[str]) -> pl.LazyFrame:
    """
    Newer Polars supports extra_columns='ignore' which helps when schemas differ slightly.
    We'll try it, and fall back if not supported.
    """
    try:
        return pl.scan_parquet(files, extra_columns="ignore")
    except TypeError:
        return pl.scan_parquet(files)


def _resolve_input_files() -> tuple[list[str], str]:
    """
    Resolve parquet files to scan based on DATE policy.
    Returns (files, chosen_date_string) where chosen_date_string can be '*' or 'YYYY-MM-DD'.
    """
    if DATE == "*":
        date_sel = "*"
    elif DATE:
        date_sel = DATE
    else:
        # Auto-pick newest date folder to avoid scanning thousands of mixed-schema files
        picked = _pick_default_date(ROOT)
        if not picked:
            raise FileNotFoundError(
                f"No date partitions found under:\n  {ROOT}\n"
                "Expected folders like date=YYYY-MM-DD.\n"
                "Fix: run the ingestors or run `python -m research.smoke_test`."
            )
        date_sel = picked
        logger.info(f"[build_features_tob] DATE not set → auto-selected latest date={date_sel}\n")

    # Support BOTH layouts inside the chosen date(s):
    # 1) partitioned: date=.../hour=.../tob_*.parquet
    # 2) flat:        date=.../tob.parquet
    patterns = [
        str(ROOT / f"date={date_sel}" / "hour=*" / "tob_*.parquet"),
        str(ROOT / f"date={date_sel}" / "tob.parquet"),
    ]

    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            "No TOB parquet files found.\n"
            f"Looked for:\n  - {patterns[0]}\n  - {patterns[1]}\n\n"
            f"Root:\n  {ROOT}\n"
        )

    # If hour-sharded files exist, ignore flat tob.parquet to avoid double-counting
    if any("\\hour=" in f or "/hour=" in f for f in files):
        files = [f for f in files if ("\\hour=" in f or "/hour=" in f)]

    return files, date_sel


def _pick_time_column(schema_names_lower: dict) -> tuple[str, str] | None:
    candidates = [
        ("ts", "auto"),
        ("ts_ms", "ms"),
        ("ts_recv_ms", "ms"),
        ("timestamp_ms", "ms"),
        ("event_time_ms", "ms"),
        ("event_ms", "ms"),
        ("e", "ms"),
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
    schema = lf.collect_schema()
    names = list(schema.keys())
    names_lower = {n.lower(): n for n in names}

    picked = _pick_time_column(names_lower)
    if not picked:
        raise KeyError(
            "Expected a time column (e.g., 'ts', 'ts_ms', 'ts_recv_ms', 'event_time_ms', etc.). "
            f"Found columns: {names}"
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
    files, date_sel = _resolve_input_files()
    logger.info(f"[load_raw_lazy] date={date_sel} | files={len(files)}\n")

    lf = _scan_parquet_compat(files)
    lf = _ensure_ts(lf)

    schema = lf.collect_schema()
    cols = {c.lower(): c for c in schema.keys()}

    def pick(*cands, default=None):
        for c in cands:
            if c in cols:
                return cols[c]
        return default

    bid_col = pick("bid", "best_bid", "b", "bid_px")
    ask_col = pick("ask", "best_ask", "a", "ask_px")
    bq_col = pick("bid_qty", "bq", "bid_size", "best_bid_qty", "bidsz", "bid_sz")
    aq_col = pick("ask_qty", "aq", "ask_size", "best_ask_qty", "asksz", "ask_sz")

    if bid_col is None or ask_col is None:
        raise KeyError(f"Could not find bid/ask columns. Found: {list(schema.keys())}")

    selects = [
        pl.col("ts"),
        pl.col(bid_col).cast(pl.Float64).alias("bid"),
        pl.col(ask_col).cast(pl.Float64).alias("ask"),
        pl.col(bq_col).cast(pl.Float64).alias("bid_qty") if bq_col else pl.lit(0.0).cast(pl.Float64).alias("bid_qty"),
        pl.col(aq_col).cast(pl.Float64).alias("ask_qty") if aq_col else pl.lit(0.0).cast(pl.Float64).alias("ask_qty"),
    ]
    return lf.select(selects)


def resample_raw_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    agg = (
        lf.group_by_dynamic(
            index_column="ts",
            every=EVERY,
            period=EVERY,
            closed="right",
            label="right",  # avoids “previous day” labels at bucket edges
        )
        .agg(
            [
                pl.col("bid").last().alias("bid"),
                pl.col("ask").last().alias("ask"),
                pl.col("bid_qty").last().alias("bid_qty"),
                pl.col("ask_qty").last().alias("ask_qty"),
            ]
        )
    )

    agg = agg.with_columns(((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"))

    agg = agg.with_columns(
        [
            (((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2.0)) * 1e4).alias("spread_bps"),
            ((pl.col("bid_qty") - pl.col("ask_qty")) / (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9)).alias("imb"),
            (
                ((pl.col("ask") * pl.col("bid_qty")) + (pl.col("bid") * pl.col("ask_qty")))
                / (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9)
            ).alias("microprice"),
        ]
    )
    return agg


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    ret_col = f"ret_{FWD_SECS}s"
    dir_col = f"dir_{FWD_SECS}s"

    df = df.with_columns((pl.col("mid").shift(-H_FWD) / pl.col("mid") - 1.0).alias(ret_col)).with_columns(
        pl.when(pl.col(ret_col) > 0)
        .then(pl.lit(1))
        .when(pl.col(ret_col) < 0)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias(dir_col)
    )

    steps_1s = max(1, 1000 // MS_INT)
    df = df.with_columns((pl.col("mid").shift(-steps_1s) / pl.col("mid") - 1.0).alias("ret_1s")).with_columns(
        pl.when(pl.col("ret_1s") > 0)
        .then(pl.lit(1))
        .when(pl.col("ret_1s") < 0)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("dir_1s")
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
    logger.info(f"Searching TOB under:\n  {ROOT}\n")

    lf_raw = load_raw_lazy()
    lf_resampled = resample_raw_lazy(lf_raw)
    df = lf_resampled.collect()

    df = add_returns(df)
    df = add_date(df)
    write_by_date(df)

    head_cols = ["ts", "mid", "spread_bps", "imb", "microprice", f"ret_{FWD_SECS}s"]
    print(df.select(head_cols).head(10))


if __name__ == "__main__":
    main()
