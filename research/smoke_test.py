# research/smoke_test.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
import os
import subprocess
import sys

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


def make_synth_tob(n: int = 2_000, dt_ms: int = 200) -> pl.DataFrame:
    """Synthetic top-of-book stream: ts, bid, ask, bid_qty, ask_qty."""
    start = dt.datetime(2099, 1, 1, 0, 0, 0)
    end = start + dt.timedelta(milliseconds=(n - 1) * dt_ms)
    ts = dt_range(start, end, dt_ms)

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


def make_synth_trades(start_dt: dt.datetime, end_dt: dt.datetime, n: int = 800) -> pl.DataFrame:
    """Synthetic aggTrade-like stream matching build_features_tob_trades expectations."""
    rng = np.random.default_rng(7)

    # Pick random times inside [start, end]
    start_utc = start_dt.replace(tzinfo=dt.timezone.utc)
    end_utc = end_dt.replace(tzinfo=dt.timezone.utc)

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    ts_trade_ms = rng.integers(low=start_ms, high=end_ms, size=n, dtype=np.int64)
    ts_trade_ms.sort()

    notional_usdt = np.clip(rng.lognormal(mean=3.0, sigma=0.8, size=n), 5.0, 2_000.0)
    is_buyer_maker = rng.random(size=n) < 0.5

    df = pl.DataFrame(
        {
            "ts_trade_ms": ts_trade_ms,
            "notional_usdt": notional_usdt,
            "is_buyer_maker": is_buyer_maker,
        }
    )
    return df


def write_partitioned(df_tob: pl.DataFrame, df_trades: pl.DataFrame) -> None:
    """Write shards in the SAME partition layout the main pipeline expects."""
    base = Path(__file__).resolve().parents[1]

    date_str = "2099-01-01"
    hour_str = "00"

    # TOB
    tob_dir_hour = (
        base
        / "data"
        / "tob"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / f"date={date_str}"
        / f"hour={hour_str}"
    )
    tob_dir_hour.mkdir(parents=True, exist_ok=True)
    tob_file = tob_dir_hour / "tob_0000.parquet"
    df_tob.write_parquet(tob_file)
    logger.info(f"→ wrote {tob_file} | rows={df_tob.height}")

    # TRADES
    trades_dir_hour = (
        base
        / "data"
        / "trades"
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / f"date={date_str}"
        / f"hour={hour_str}"
    )
    trades_dir_hour.mkdir(parents=True, exist_ok=True)
    trades_file = trades_dir_hour / "trades_0000.parquet"
    df_trades.write_parquet(trades_file)
    logger.info(f"→ wrote {trades_file} | rows={df_trades.height}")


def run(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd))


def main():
    logger.info("alpha-exec-stack offline smoke test")

    # Defaults keep CI fast; you can override in env
    resample_ms = int(os.getenv("RESAMPLE_MS", "200"))
    fwd_secs = int(os.getenv("FWD_SECS", "2"))

    # 1) synth data
    df_tob = make_synth_tob(n=2_000, dt_ms=resample_ms)
    start_dt = df_tob.select(pl.col("ts").min()).item()
    end_dt = df_tob.select(pl.col("ts").max()).item()
    df_trades = make_synth_trades(start_dt, end_dt, n=800)

    # 2) write partitioned shards
    write_partitioned(df_tob, df_trades)

    # 3) run the minimal end-to-end pipeline (fast grid)
    env = os.environ.copy()
    env.setdefault("RESAMPLE_MS", str(resample_ms))
    env.setdefault("FWD_SECS", str(fwd_secs))

    # keep smoke fast
    env.setdefault("K_FOLDS", os.getenv("K_FOLDS", "2"))
    env.setdefault("THRESH_GRID", os.getenv("THRESH_GRID", "0.60"))
    env.setdefault("EDGE_GRID", os.getenv("EDGE_GRID", "0.00"))
    env.setdefault("FILL_USDT", os.getenv("FILL_USDT", "25"))

    py = sys.executable
    base = Path(__file__).resolve().parents[1]


    logger.info("→ build features (TOB)")
    run([py, str(base / "research" / "build_features_tob.py")], env=env, cwd=base)

    logger.info("→ join TOB with trades")
    run([py, str(base / "research" / "build_features_tob_trades.py")], env=env, cwd=base)

    logger.info("→ taker evaluation (GBM)")
    run([py, str(base / "research" / "train_tob_gbm.py")], env=env, cwd=base)

    logger.info("→ maker evaluation")
    run([py, str(base / "research" / "train_tob_maker.py")], env=env, cwd=base)

    logger.info("✅ smoke test complete (see ./results/metrics and ./results/plots)")


if __name__ == "__main__":
    main()