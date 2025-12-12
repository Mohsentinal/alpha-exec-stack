# research/smoke_test.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger

# ---- tiny compat: works on both old (low/high) and new (start/end) Polars
def dt_range(start_dt: dt.datetime, end_dt: dt.datetime, interval_ms: int) -> pl.Series:
    try:
        # Polars >= 0.20: start/end
        return pl.datetime_range(
            start=start_dt,
            end=end_dt,
            interval=f"{interval_ms}ms",
            eager=True,
            time_unit="ms",
        )
    except TypeError:
        # Older Polars: low/high
        return pl.datetime_range(
            low=start_dt,
            high=end_dt,
            interval=f"{interval_ms}ms",
            eager=True,
            time_unit="ms",
        )

def make_synth(n: int = 2_000, dt_ms: int = 200) -> pl.DataFrame:
    """
    Build a tiny synthetic top-of-book stream suitable for quick pipeline checks.
    Columns: ts, bid, ask, bid_qty, ask_qty
    """
    start = dt.datetime(2099, 1, 1, 0, 0, 0)
    end = start + dt.timedelta(milliseconds=(n - 1) * dt_ms)
    ts = dt_range(start, end, dt_ms)

    # simple random walk mid with small spread and qty noise
    rng = np.random.default_rng(42)
    mid = 100_000 + np.cumsum(rng.normal(0, 1.0, n))
    spread = np.clip(rng.normal(0.8, 0.1, n), 0.4, 1.6)
    bid = mid - (spread * 0.5)
    ask = mid + (spread * 0.5)
    bid_qty = np.clip(rng.normal(5.0, 1.5, n), 0.5, None)
    ask_qty = np.clip(rng.normal(5.0, 1.5, n), 0.5, None)

    return pl.DataFrame(
        {
            "ts": ts,
            "bid": bid,
            "ask": ask,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
        }
    )

def main():
    df = make_synth()
    logger.info(f"shape: {df.shape}")
    logger.info(df.head(5))

    # Write a shard in the same partition layout your pipeline expects
    base = Path(__file__).resolve().parents[1]
    out_dir = (
        base
        / "data"
        / "tob"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "date=2099-01-01"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "tob.parquet"
    # ✅ Polars requires a path/handle here (does not return bytes)
    df.write_parquet(out_file)

    logger.info(f"→ wrote {out_file} | rows={df.height}")

if __name__ == "__main__":
    main()
