# alpha-exec-stack

A compact research pipeline that:

* streams **Binance BTCUSDT** top-of-book and trades,
* builds **time-bucketed microstructure features**,
* trains a fast **taker** classifier,
* evaluates simple **maker** entries with **fill gates** (require future opposite-side flow),
* prints **net basis-point PnL** after fees/spread.

> This repository is for learning and experimentation only. Not trading advice.

---

## How it works (end-to-end)

1. **Ingest**

   * `binance_bookticker_ingest.py` subscribes to `bookTicker` and writes Parquet shards under `data/tob/...`.
   * `binance_trades_ingest.py` subscribes to `aggTrade` and writes under `data/trades/...`.

2. **Feature engineering**

   * `build_features_tob.py` resamples the top-of-book to a fixed grid (100–200 ms) and computes:

     * `spread_bps`, `imb` (bid/ask imbalance), `ofi` (order-flow imbalance proxy), `microprice`, returns, lags.
   * `build_features_tob_trades.py` buckets trades on the same grid and adds:

     * `hit_bid_notional`, `hit_ask_notional`,
     * **forward 1s sums** `hit_*_notional_fwd1s` (used as maker fill gates).

3. **Modeling / evaluation**

   * **Taker**: `train_tob_gbm.py` (Histogram Gradient Boosting) predicts next-second direction on features.
     Purged K-fold CV avoids look-ahead. Results printed as hit-rate, trade-rate, gross/net bps.
   * **Maker**: `train_tob_maker.py` takes the same classifier and only places limit entries when:

     * model confidence ≥ threshold,
     * microprice edge ≥ threshold,
     * **future opposite-side notional ≥ $X** within ~1s (proxy for getting filled).
       Exit is assumed **taker**; costs = maker fee at entry + taker fee + ½ future spread.

---

## Repository layout

```
ingest/
  binance_bookticker_ingest.py
  binance_trades_ingest.py
research/
  build_features_tob.py
  build_features_tob_trades.py
  train_tob_gbm.py
  train_tob_maker.py
run_pipeline.py           # orchestration
requirements.txt
data/                     # generated (not committed)
logs/                     # run logs (not committed)
```

---

## Installation

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running

### One-shot pipeline

```bash
python run_pipeline.py
```

What it does: start both ingestors → wait for `DURATION_MIN` minutes (see **Config**) → build features → join with trades → train taker GBM → evaluate maker entries → print metrics.

### Individual stages (optional)

```bash
python ingest/binance_bookticker_ingest.py
python ingest/binance_trades_ingest.py
python research/build_features_tob.py
python research/build_features_tob_trades.py
python research/train_tob_gbm.py
python research/train_tob_maker.py
```

---

## Configuration

Change these small knobs in code (kept simple on purpose):

* **Sampling speed**
  `research/build_features_tob.py` → `EVERY = "100ms"` or `"200ms"` (default 200 ms).
  `research/build_features_tob_trades.py` and `research/train_tob_maker.py` should match the same `EVERY`.

* **Data scope**
  `EXCHANGE = "binance"`, `SYMBOL = "BTCUSDT"` (files are partitioned by these).

* **Runner timing**
  `run_pipeline.py` → `DURATION_MIN = 30` (how long to collect before training).

* **Fees**
  `research/train_tob_maker.py` → `MAKER_FEE_BPS`, `TAKER_FEE_BPS`.

* **Maker gates**
  `THRESH_GRID`, `EDGE_GRID`, `FILL_USDT` in `train_tob_maker.py`.
  Increase `FILL_USDT` to demand stronger future aggressor flow (fewer but higher-quality fills).

---

## Outputs

* **Features**
  `data/features_tob/.../features_tob_resample-<100ms|200ms>.parquet`
  `data/features_tobtrades/.../features_tobtrades_resample-<100ms|200ms>.parquet`

* **Console metrics (examples)**

```
[thr=0.70, edge≥0.00bps] hit=0.715 | trade_rate=0.046 | avg_gross_bps=0.46 | avg_net_bps=-9.54
[thr=0.70, edge≥0.00bps, fill≥$50] hit=0.309 | trade_rate=0.005 | avg_gross_bps=0.04 | avg_net_bps=-1.16
```

Interpretation:

* **hit**: fraction of correct directions when a trade was taken.
* **trade_rate**: how often the strategy acts.
* **avg_gross_bps / avg_net_bps**: average per-trade return before/after costs (basis points; 1 bp = 0.01%).

If you see `nan` hit for some maker rows, the gates were too strict for the current dataset (no qualifying trades).

---

## Method details (concise)

* **Labels**: next-second mid-price return (`ret_1s`), direction `dir_1s ∈ {−1, 0, +1}`.
* **Features**: `imb`, `ofi`, `micro_edge_bps`, deltas, rolling sums, short return lags, `spread_bps`.
* **CV**: purged K-fold with an embargo window aligned to the resample step.
* **Costs**:

  * taker: fee + ½ current spread,
  * maker: entry as maker (no entry spread) + exit as taker (fee + ½ future spread).

---

## Limitations

* Uses public feeds only; no depth ladder beyond best bid/ask.
* Fill modeling is heuristic (forward notional as a proxy).
* Assumes fixed fees and simple exits; no inventory, queue priority, or latency effects.
* Results are **not** tradeable PnL; they’re indicators for whether a signal survives basic costs.

---

## Roadmap / ideas

* Add deeper book features (levels 2–10).
* Replace heuristic fills with simulated order-book matching.
* Try calibrated probabilities and cost-aware thresholds.
* Log feature importances and SHAP summaries.
* Add symbol configurability and multi-day backfills.

---

## Requirements

See `requirements.txt`. Core libs: `websockets`, `orjson`, `polars`, `pyarrow`, `numpy`, `scikit-learn`, `lightgbm`, `tenacity`, `loguru`, `python-dateutil`.

---

## License

MIT
