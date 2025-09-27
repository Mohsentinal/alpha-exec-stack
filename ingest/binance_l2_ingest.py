import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone
import signal
import sys

import websockets
import orjson
import polars as pl
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger

# ----------------------------
# Config (edit these safely)
# ----------------------------
EXCHANGE = "binance"
SYMBOL = "btcusdt"              # lowercase, e.g., 'btcusdt', 'ethusdt'
STREAM_SUFFIX = "@depth@100ms"  # depth diffs; alternative: "@depth5@100ms"
WS_URL = "wss://stream.binance.com:9443/ws"

# Anchor OUTPUT_ROOT to the repo root, regardless of working directory
BASE_DIR = Path(__file__).resolve().parents[1]     # .../alpha-exec-stack
OUTPUT_ROOT = BASE_DIR / "data" / "quotes"         # partitioned parquet output root

# Flush policy
FLUSH_EVERY_ROWS = 1000         # flush when buffer reaches this size
FLUSH_EVERY_SECONDS = 5.0       # flush at least this often

# Logging
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    enqueue=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
)
logger.info(f"Writing parquet under: {OUTPUT_ROOT}")

# Internal state
_buffer = []
_last_flush_ts = None
_stop = asyncio.Event()

def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _partition_dir(ts_ms: int, symbol: str) -> Path:
    """data/quotes/exchange=binance/symbol=BTCUSDT/date=YYYY-MM-DD/hour=HH (UTC)"""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return (
        OUTPUT_ROOT
        / f"exchange={EXCHANGE}"
        / f"symbol={symbol.upper()}"
        / f"date={dt.strftime('%Y-%m-%d')}"
        / f"hour={dt.strftime('%H')}"
    )

def _flush_buffer(force: bool = False):
    """Write buffered rows to a UTC-partitioned Parquet file."""
    global _buffer, _last_flush_ts
    if not _buffer:
        return

    now_ms = _now_utc_ms()
    should_flush_time = (_last_flush_ts is None) or (
        (now_ms - _last_flush_ts) >= FLUSH_EVERY_SECONDS * 1000
    )
    should_flush_rows = len(_buffer) >= FLUSH_EVERY_ROWS
    if not (force or should_flush_time or should_flush_rows):
        return

    df = pl.from_dicts(_buffer).with_columns(
        [
            pl.col("ts_event_ms").cast(pl.Int64),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("u").cast(pl.Int64, strict=False),
            pl.col("pu").cast(pl.Int64, strict=False),
            pl.col("side").cast(pl.Categorical),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("exchange").cast(pl.Utf8),
        ]
    )

    if len(df) > 0:
        last_ts_ms = int(df[-1, "ts_event_ms"])
        part_dir = _partition_dir(last_ts_ms, df[-1, "symbol"])
        part_dir.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        fname = part_dir / f"l2_{now_utc.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.parquet"
        df.write_parquet(fname)
        logger.info(f"Flushed {len(df):,} rows → {fname}")

    _buffer = []
    _last_flush_ts = now_ms

def _handle_depth_message(msg: dict, symbol: str):
    """Normalize Binance depth diff message into flat rows."""
    ts_event_ms = int(msg.get("E", _now_utc_ms()))
    u = msg.get("u")
    pu = msg.get("U")
    for p, q in msg.get("b", []):
        _buffer.append({"ts_event_ms": ts_event_ms, "exchange": EXCHANGE, "symbol": symbol.upper(),
                        "side": "bid", "price": float(p), "size": float(q), "u": u, "pu": pu})
    for p, q in msg.get("a", []):
        _buffer.append({"ts_event_ms": ts_event_ms, "exchange": EXCHANGE, "symbol": symbol.upper(),
                        "side": "ask", "price": float(p), "size": float(q), "u": u, "pu": pu})

async def _consumer(ws, symbol: str):
    global _last_flush_ts
    _last_flush_ts = _now_utc_ms()
    async for raw in ws:
        try:
            msg = orjson.loads(raw)
        except Exception:
            logger.exception("JSON decode error")
            continue
        if msg.get("e") == "depthUpdate":
            _handle_depth_message(msg, symbol)
        _flush_buffer(False)
        if _stop.is_set():
            break

@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(100))
async def _connect_and_run(symbol: str):
    stream = f"{symbol}{STREAM_SUFFIX}"
    url = f"{WS_URL}/{stream}"
    logger.info(f"Connecting: {url}")
    async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=2000) as ws:
        logger.info("Connected.")
        await _consumer(ws, symbol)
    logger.warning("WebSocket closed, will retry...")

async def main():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop.set)
        except NotImplementedError:
            pass
    try:
        await _connect_and_run(SYMBOL)
    finally:
        _flush_buffer(True)
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _stop.set()
        _flush_buffer(True)
        logger.info("Exited by user.")
