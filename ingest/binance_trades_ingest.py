import asyncio, os, sys, signal
from pathlib import Path
from datetime import datetime, timezone

import websockets, orjson, polars as pl
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger

EXCHANGE = "binance"
SYMBOL = "btcusdt"                      # lowercase
WS_URL = "wss://stream.binance.com:9443/ws"
STREAM_SUFFIX = "@aggTrade"

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / "data" / "trades" / f"exchange={EXCHANGE}" / f"symbol={SYMBOL.upper()}"

FLUSH_EVERY_ROWS = 3000
FLUSH_EVERY_SECONDS = 5.0

logger.remove()
logger.add(sys.stdout, level="INFO", enqueue=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
logger.info(f"Writing trades parquet under: {OUTPUT_ROOT}")

_buffer, _last_flush_ts = [], None
_stop = asyncio.Event()

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def _part_dir(ts_ms: int) -> Path:
    dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
    return OUTPUT_ROOT / f"date={dt.strftime('%Y-%m-%d')}" / f"hour={dt.strftime('%H')}"

def _flush(force=False):
    global _buffer, _last_flush_ts
    if not _buffer: return
    now = _now_ms()
    by_time = (_last_flush_ts is None) or ((now - _last_flush_ts) >= FLUSH_EVERY_SECONDS*1000)
    by_rows = len(_buffer) >= FLUSH_EVERY_ROWS
    if not (force or by_time or by_rows): return

    df = pl.from_dicts(_buffer).with_columns([
        pl.col("ts_event_ms").cast(pl.Int64),
        pl.col("ts_trade_ms").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
        (pl.col("price")*pl.col("qty")).alias("notional_usdt").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Boolean),
        pl.lit(EXCHANGE).alias("exchange"),
        pl.lit(SYMBOL.upper()).alias("symbol"),
    ])
    last_ts = int(df[-1, "ts_event_ms"])
    d = _part_dir(last_ts)
    d.mkdir(parents=True, exist_ok=True)
    fname = d / f"trades_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.parquet"
    df.write_parquet(fname)
    logger.info(f"Flushed {len(df):,} rows → {fname}")
    _buffer.clear(); _last_flush_ts = now

def _handle(msg: dict):
    # docs: https://binance-docs.github.io/apidocs/spot/en/#aggregate-trade-streams
    # keys: E(event time), T(trade time), p(price), q(qty), m(is buyer the maker)
    _buffer.append({
        "ts_event_ms": int(msg.get("E")),
        "ts_trade_ms": int(msg.get("T")),
        "price": float(msg["p"]),
        "qty": float(msg["q"]),
        "is_buyer_maker": bool(msg["m"]),
    })

async def _consumer(ws):
    global _last_flush_ts
    _last_flush_ts = _now_ms()
    async for raw in ws:
        try:
            msg = orjson.loads(raw)
        except Exception:
            logger.exception("JSON decode error"); continue
        if all(k in msg for k in ("p","q","T","E","m")):
            _handle(msg)
        _flush(False)
        if _stop.is_set(): break

@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(100))
async def _run():
    url = f"{WS_URL}/{SYMBOL}{STREAM_SUFFIX}"
    logger.info(f"Connecting: {url}")
    async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=2000) as ws:
        logger.info("Connected.")
        await _consumer(ws)
    logger.warning("WebSocket closed, will retry...")

async def main():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, _stop.set)
        except NotImplementedError: pass
    try:
        await _run()
    finally:
        _flush(True); logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _stop.set(); _flush(True); logger.info("Exited by user.")
