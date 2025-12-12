# alpha-exec-stack

A small, end-to-end research pipeline for **microstructure / limit-order-book signals** on **Binance BTCUSDT**.

It does three things in one repo:

1) **Collect** real-time top-of-book + trades (websocket → Parquet, partitioned by date)  
2) **Build** sub-second features on a fixed grid (default **200ms**) and create forward labels (default **2s**)  
3) **Evaluate** two simple decision styles:
   - **taker** (market/agg) decisions from a fast classifier  
   - **maker** (limit) entries gated by a fill proxy + spread/fee costs

> Learning + portfolio project. Not trading advice.

---

## Why this repo exists

I wanted a project that is **actually runnable**, not just notebooks or screenshots:
- real-time ingestion
- columnar feature pipeline
- clear labeling for forward prediction
- basic execution-aware evaluation
- reproducibility (smoke test + CI)

This is meant to be a clean “starter stack” I can extend into deeper execution simulation later.

---

## What’s inside

### Ingest (real-time → Parquet)
- `ingest/binance_bookticker_ingest.py`  
  Subscribes to **bookTicker** (best bid/ask + sizes), writes shards under `data/tob/...`
- `ingest/binance_trades_ingest.py`  
  Subscribes to **aggTrade**, writes shards under `data/trades/...`

### Features (fixed grid)
- `research/build_features_tob.py`  
  Resamples TOB to `RESAMPLE_MS` (default `200ms`) and builds:
  - `mid`, `spread_bps`, `imb`, `microprice`
  - `ofi` proxy (from bid/ask size deltas), rolling OFI sums
  - short lags/deltas
  - forward returns `ret_{k}s` and direction labels `dir_{k}s` (k = `FWD_SECS`)

- `research/build_features_tob_trades.py`  
  Buckets trades on the same grid and adds:
  - `hit_bid_notional`, `hit_ask_notional`
  - forward sums like `hit_*_notional_fwd1s`, `hit_*_notional_fwd2s`, ...  
    (used as a **maker fill gate** proxy)

### Models / evaluation
- `research/train_tob_gbm.py`  
  Histogram Gradient Boosting classifier (fast, simple) on TOB features.  
  Prints a small grid over:
  - probability threshold (`THRESH_GRID`)
  - micro-edge gate (`EDGE_GRID`)
  
  Outputs CSV to `results/metrics/`.

- `research/train_tob_maker.py`  
  Uses the same classifier signal, but evaluates **maker entries** only when:
  - model confidence ≥ threshold  
  - `|micro_edge_bps|` ≥ threshold  
  - **future opposite-side aggressive notional ≥ $X** within the forward window  
    (proxy for “would I plausibly get filled?”)
  
  Costs include **maker + taker fees** and **½ future spread** at exit.  
  Outputs CSV to `results/metrics/`.

### Orchestration
- `run_pipeline.py`  
  Starts both ingestors → waits → builds features → joins trades → runs both evaluations.

---

## Quick start

```bash
python -m venv .venv
# Windows:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
