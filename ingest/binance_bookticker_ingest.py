import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone
import signal, sys

import websockets, orjson, polars as pl
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger

# --- Config ---
EXCHANGE = "binance"
SYMBOL = "btcusdt"                 # lowercase, e.g., 'btcusdt'
WS_URL = "wss://stream.binance.com:9443/ws"
STREAM_SUFFIX = "@bookTicker"      # best bid/ask + sizes

# Anchor to repo root
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / "data" / "tob"  # top-of-book parquet

FLUSH_EVERY_ROWS = 2000
FLUSH_EVERY_SECONDS = 5.0

logger.remove()
logger.add(sys.stdout, level="INFO", enqueue=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
logger.info(f"Writing TOB parquet under: {OUTPUT_ROOT}")

_buffer = []
_last_flush_ts = None
_stop = asyncio.Event()

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _part_dir(ts_ms: int, symbol: str) -> Path:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return (OUTPUT_ROOT /
            f"exchange={EXCHANGE}" /
            f"symbol={symbol.upper()}" /
            f"date={dt.strftime('%Y-%m-%d')}" /
            f"hour={dt.strftime('%H')}")

def _flush(force: bool = False):
    global _buffer, _last_flush_ts
    if not _buffer: return
    now = _now_ms()
    by_time = (_last_flush_ts is None) or ((now - _last_flush_ts) >= FLUSH_EVERY_SECONDS * 1000)
    by_rows = len(_buffer) >= FLUSH_EVERY_ROWS
    if not (force or by_time or by_rows): return

    df = pl.from_dicts(_buffer).with_columns([
        pl.col("ts_recv_ms").cast(pl.Int64),
        pl.col("bid_px").cast(pl.Float64),
        pl.col("bid_sz").cast(pl.Float64),
        pl.col("ask_px").cast(pl.Float64),
        pl.col("ask_sz").cast(pl.Float64),
        pl.col("u").cast(pl.Int64, strict=False),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("exchange").cast(pl.Utf8),
    ])
    if len(df) > 0:
        last_ts = int(df[-1, "ts_recv_ms"])
        d = _part_dir(last_ts, df[-1, "symbol"])
        d.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        fname = d / f"tob_{now_utc.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.parquet"
        df.write_parquet(fname)
        logger.info(f"Flushed {len(df):,} rows → {fname}")
    _buffer = []
    _last_flush_ts = now

def _handle_bookticker(msg: dict, symbol: str):
    # Sample fields: { "u": 123, "s": "BTCUSDT", "b": "price", "B": "qty", "a": "price", "A": "qty" }
    ts_recv_ms = _now_ms()  # bookTicker has no E (event time), so use receive time consistently
    _buffer.append({
        "ts_recv_ms": ts_recv_ms,
        "exchange": EXCHANGE,
        "symbol": symbol.upper(),
        "u": msg.get("u"),
        "bid_px": float(msg["b"]),
        "bid_sz": float(msg["B"]),
        "ask_px": float(msg["a"]),
        "ask_sz": float(msg["A"]),
    })

async def _consumer(ws, symbol: str):
    global _last_flush_ts
    _last_flush_ts = _now_ms()
    async for raw in ws:
        try:
            msg = orjson.loads(raw)
        except Exception:
            logger.exception("JSON decode error")
            continue
        # bookTicker messages don’t include 'e'; rely on presence of keys
        if all(k in msg for k in ("b","B","a","A")):
            _handle_bookticker(msg, symbol)
        _flush(False)
        if _stop.is_set():
            break

@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(100))
async def _connect_and_run(symbol: str):
    url = f"{WS_URL}/{symbol}{STREAM_SUFFIX}"
    logger.info(f"Connecting: {url}")
    async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=2000) as ws:
        logger.info("Connected.")
        await _consumer(ws, symbol)
    logger.warning("WebSocket closed, will retry...")

async def main():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, _stop.set)
        except NotImplementedError: pass
    try:
        await _connect_and_run(SYMBOL)
    finally:
        _flush(True)
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _stop.set(); _flush(True)
        logger.info("Exited by user.")
