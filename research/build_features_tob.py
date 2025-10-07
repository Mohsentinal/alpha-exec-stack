# research/build_features_tob.py
from __future__ import annotations

from pathlib import Path
import os
import re

import polars as pl
from loguru import logger


# ============
# Config
# ============

# Exchange/symbol (allow env overrides, keep your defaults)
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL   = os.getenv("SYMBOL", "BTCUSDT")

# Accept RESAMPLE_MS as "200ms" (preferred) or a bare integer "200" (we normalize -> "200ms")
_every_raw = os.getenv("RESAMPLE_MS", "200ms").strip()

if re.fullmatch(r"\d+", _every_raw):               # e.g. "200"
    EVERY = f"{_every_raw}ms"
elif re.fullmatch(r"\d+\s*ms", _every_raw):        # e.g. "200ms" or "200 ms"
    EVERY = re.sub(r"\s+", "", _every_raw)
else:
    raise ValueError(f"RESAMPLE_MS must look like '200ms' (got '{_every_raw}')")

# numeric milliseconds (e.g. 200 from "200ms")
MS = int(re.match(r"(\d+)", EVERY).group(1))

# Forward horizon in seconds (used to build ret_1s/dir_1s labels; keep the column name for downstream code)
FWD_SECS = int(os.getenv("FWD_SECS", "1"))
H_FWD = max(1, round((1000 * FWD_SECS) / MS))     # number of bars in ~FWD_SECS seconds

# Paths
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"
OUT = BASE / "data" / "features_tob" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}"

logger.remove()
logger.add(lambda m: print(m, end=""))
logger.info(f"[build_features_tob] using EVERY='{EVERY}' (MS={MS}) | FWD_SECS={FWD_SECS} | H_FWD={H_FWD}\n")


# ============
# Helpers
# ============

def find_files() -> list[str]:
    """Find all raw TOB parquet chunks under date/hour partitions."""
    print("2025 | INFO     | __main__:find_files:23 - Searching TOB under:\n"
          f"  {RAW}")
    files = sorted(str(p) for p in RAW.glob("date=*/*/tob_*.parquet"))
    print(f"Found {len(files)} files.\n")
    return files


def _canon_name(cols: list[str], candidates: list[str], required: bool = True) -> str | None:
    """Return the first present column name among candidates."""
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in columns: {cols}")
    return None


def load_raw_lazy() -> pl.LazyFrame:
    """
    Load raw book-ticker as a LazyFrame.
    Aim to be schema-tolerant across slightly different ingestor versions.
    Expected canonical outputs after select+alias: ts(datetime[ms]), bid, ask, bid_sz, ask_sz.
    """
    lf = pl.scan_parquet(str(RAW / "date=*" / "hour=*" / "tob_*.parquet"))

    cols = lf.columns

    # Time column: prefer 'ts' (datetime). If only 'ts_ms' exists, convert to datetime.
    ts_col = "ts" if "ts" in cols else None
    ts_ms_col = "ts_ms" if "ts_ms" in cols else None
    if not ts_col and not ts_ms_col:
        raise KeyError("Expected a time column 'ts' or 'ts_ms' in raw TOB files.")

    # Price & size columns: try common variants
    bid_col     = _canon_name(cols, ["bid", "best_bid", "bestBid"])
    ask_col     = _canon_name(cols, ["ask", "best_ask", "bestAsk"])
    bid_sz_col  = _canon_name(cols, ["bid_sz", "bid_size", "best_bid_qty", "bidQty"])
    ask_sz_col  = _canon_name(cols, ["ask_sz", "ask_size", "best_ask_qty", "askQty"])

    exprs = []

    # ts
    if ts_col:
        exprs.append(pl.col(ts_col).alias("ts"))
    else:
        exprs.append(pl.from_epoch(pl.col(ts_ms_col), time_unit="ms").alias("ts"))

    # prices/sizes (cast to float)
    exprs.extend([
        pl.col(bid_col).cast(pl.Float64).alias("bid"),
        pl.col(ask_col).cast(pl.Float64).alias("ask"),
        pl.col(bid_sz_col).cast(pl.Float64).alias("bid_sz"),
        pl.col(ask_sz_col).cast(pl.Float64).alias("ask_sz"),
    ])

    return lf.select(exprs)


def resample_raw_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Resample ticks to a regular grid using group_by_dynamic on 'ts'.
    We take LAST price/size per bar to represent the top-of-book at bar close.
    """
    return (
        lf.group_by_dynamic(index_column="ts", every=EVERY, period=EVERY, closed="right")
          .agg([
              pl.col("bid").last(),
              pl.col("ask").last(),
              pl.col("bid_sz").last(),
              pl.col("ask_sz").last(),
          ])
          .sort("ts")
    )


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute mid, spread_bps, imbalance, OFI, microprice, forward return/dir,
    and a string 'date' partition column. Keep names expected downstream:
    - ret_1s / dir_1s are computed over FWD_SECS, but we keep the column names for compatibility.
    """
    # base metrics
    df = df.with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
    ])
    df = df.with_columns([
        (((pl.col("ask") - pl.col("bid")) / pl.col("mid")) * 1e4).alias("spread_bps")
    ])

    # imbalance
    denom = (pl.col("bid_sz") + pl.col("ask_sz"))
    df = df.with_columns([
        pl.when(denom.abs() > 0)
          .then((pl.col("bid_sz") - pl.col("ask_sz")) / denom)
          .otherwise(0.0)
          .alias("imb")
    ])

    # simple OFI (order flow imbalance at top of book)
    bid_t, bid_p = pl.col("bid"), pl.col("bid").shift(1)
    ask_t, ask_p = pl.col("ask"), pl.col("ask").shift(1)
    bs_t, bs_p   = pl.col("bid_sz"), pl.col("bid_sz").shift(1)
    as_t, as_p   = pl.col("ask_sz"), pl.col("ask_sz").shift(1)

    ofi_bid = (
        pl.when(bid_t > bid_p).then(bs_t)
         .when(bid_t < bid_p).then(-bs_p)
         .otherwise(0.0)
    )
    ofi_ask = (
        pl.when(ask_t < ask_p).then(as_t)
         .when(ask_t > ask_p).then(-as_p)
         .otherwise(0.0)
    )
    df = df.with_columns((ofi_bid + ofi_ask).alias("ofi"))

    # microprice (size-weighted)
    den_mp = (pl.col("bid_sz") + pl.col("ask_sz"))
    df = df.with_columns([
        pl.when(den_mp.abs() > 0)
          .then((pl.col("bid") * pl.col("ask_sz") + pl.col("ask") * pl.col("bid_sz")) / den_mp)
          .otherwise(pl.col("mid"))
          .alias("microprice")
    ])

    # forward return over ~FWD_SECS seconds (kept as ret_1s/dir_1s for compatibility)
    df = df.with_columns([
        (pl.col("mid").shift(-H_FWD) / pl.col("mid") - 1.0).alias("ret_1s"),
    ])
    df = df.with_columns([
        pl.when(pl.col("ret_1s") > 0).then(pl.lit(1))
         .when(pl.col("ret_1s") < 0).then(pl.lit(-1))
         .otherwise(pl.lit(0))
         .alias("dir_1s")
    ])

    # date partition
    df = df.with_columns([
        pl.col("ts").dt.date().cast(pl.Utf8).alias("date")
    ])

    # order final columns
    cols = ["ts", "date", "bid", "ask", "bid_sz", "ask_sz",
            "mid", "spread_bps", "imb", "ofi", "microprice", "ret_1s", "dir_1s"]
    return df.select([c for c in cols if c in df.columns])


def write_features(df: pl.DataFrame) -> None:
    """
    Write one parquet per date under:
      data/features_tob/exchange=.../symbol=.../date=YYYY-MM-DD/features_tob_resample-{EVERY}.parquet
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for d in df.select("date").unique().to_series().to_list():
        sub = df.filter(pl.col("date") == d)
        out_dir = OUT / f"date={d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"features_tob_resample-{EVERY}.parquet"
        sub.write_parquet(out_path)
        logger.info(f"→ wrote {out_path}\n")


def main():
    files = find_files()
    if not files:
        raise FileNotFoundError(f"No raw TOB parquet files found under {RAW}")

    lf_raw = load_raw_lazy()
    lf_resampled = resample_raw_lazy(lf_raw)
    df = lf_resampled.collect(streaming=True)
    df_feat = build_features(df)
    write_features(df_feat)

    # small preview for the console
    print(
        df_feat.select("ts", "mid", "spread_bps", "imb", "ofi", "microprice", "ret_1s")
               .head(10)
               .to_string()
    )


if __name__ == "__main__":
    main()
