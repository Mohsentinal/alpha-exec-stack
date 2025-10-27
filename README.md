alpha-exec-stack
===================

A small, real-time research pipeline for limit-order-book (LOB) signals.
It ingests Binance BTCUSDT data, builds microstructure features on a fixed time grid,
trains a fast classifier for taker decisions, and evaluates simple maker entries with
realistic fill gates and costs.

NOTE: For learning and experimentation only. Not trading advice.

---------------------------------------------------------------------
WHY THIS EXISTS (AND WHAT IT SHOWS ABOUT ME)
---------------------------------------------------------------------
• Stream & persist market data reliably (websocket → Parquet, partitioned on disk)
• Engineer microstructure features on sub-second grids (efficient columnar tooling)
• Train & evaluate models with careful labeling and basic cost modeling
• Defensive code: schema drift, timestamp quirks, NaN guards, logging
• Clear end-to-end pipeline you can run in one command

---------------------------------------------------------------------
WHAT THE PIPELINE DOES
---------------------------------------------------------------------
1) Ingest
   - ingest/binance_bookticker_ingest.py → subscribes to bookTicker, writes data/tob/...
   - ingest/binance_trades_ingest.py   → subscribes to aggTrade,   writes data/trades/...

2) Feature engineering (fixed 200 ms grid by default)
   - research/build_features_tob.py:
       mid, spread_bps, imb (imbalance), ofi (from bid/ask size deltas),
       microprice, forward returns ret_{k}s, labels dir_{k}s, short lags & deltas.
   - research/build_features_tob_trades.py:
       hit_bid_notional, hit_ask_notional,
       forward sums (hit_*_notional_fwd1s/fwd2s/fwd5s) used as maker fill gates.

3) Modeling & evaluation
   - Taker: research/train_tob_gbm.py (Histogram Gradient Boosting) predicts dir_{k}s (default k=2s);
            prints hit-rate, trade-rate, and gross/net bps after taker fees.
   - Maker: research/train_tob_maker.py places limit entries only when:
            * model confidence ≥ threshold
            * |micro_edge_bps| ≥ threshold
            * future opposite-side notional ≥ $X within ~1s (proxy for getting filled)
            Exit assumes taker; costs include maker + taker fees + ½ future spread.

4) Orchestration
   - run_pipeline.py: start ingestors → wait → build features → join trades
                      → train taker model → evaluate maker entries → print metrics.

---------------------------------------------------------------------
QUICK START
---------------------------------------------------------------------
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

Run end-to-end (default: 200 ms grid, 2-second labels):
python run_pipeline.py

---------------------------------------------------------------------
CONFIGURATION (SMALL, EXPLICIT KNOBS)
---------------------------------------------------------------------
Environment variables (defaults shown):
  EXCHANGE      = binance
  SYMBOL        = BTCUSDT
  RESAMPLE_MS   = 200ms   (also accepts bare integers like 200)
  FWD_SECS      = 2
  TAKER_FEE_BPS = 5.0
  MAKER_FEE_BPS = 1.0

Windows PowerShell:
  $env:RESAMPLE_MS = "200ms"
  $env:FWD_SECS    = "2"

bash:
  export RESAMPLE_MS=200ms
  export FWD_SECS=2

---------------------------------------------------------------------
EXAMPLE OUTPUT (ABRIDGED)
---------------------------------------------------------------------
Taker (GBM)
[thr=0.65, edge≥0.05bps] hit=0.87 | trade_rate=0.001 | avg_gross_bps=1.62 | avg_net_bps=-3.38

Maker (fill-gated)
[thr=0.70, edge≥0.01bps, fill≥$50] hit=0.812 | trade_rate=0.000 | avg_gross_bps=0.35 | avg_net_bps=-2.05

How to read:
  hit        = fraction of correct directions when a trade was taken
  trade_rate = how often the signal triggers
  avg_*_bps  = average per-trade return before/after costs (1 bp = 0.01%)
  If some maker rows say "nan", gates were too strict (no qualifying trades).

---------------------------------------------------------------------
REPOSITORY LAYOUT
---------------------------------------------------------------------
ingest/
  binance_bookticker_ingest.py
  binance_trades_ingest.py
research/
  build_features_tob.py
  build_features_tob_trades.py
  train_tob_gbm.py
  train_tob_maker.py
run_pipeline.py
requirements.txt
data/    (generated, ignored)
logs/    (generated, ignored)

---------------------------------------------------------------------
WHAT I LEARNED / SKILLS DEMONSTRATED
---------------------------------------------------------------------
• Streaming data engineering (websockets → Parquet, schema drift handling)
• Microstructure features (imbalance, microprice, OFI from size deltas)
• Sub-second resampling; forward returns & clean labels on fixed grids
• Purged splits to reduce look-ahead; simple, fast GBM
• Execution-aware evaluation (maker fill gates, fee & spread costs)
• Reproducible pipeline with logging and NaN guards

---------------------------------------------------------------------
LIMITATIONS (BY DESIGN)
---------------------------------------------------------------------
• Public feeds only; no full depth ladder or queue position modeling
• Fill modeling is heuristic (forward notional proxy)
• Costs simplified; no inventory/latency simulation
• Metrics are not tradeable PnL; they’re research indicators

---------------------------------------------------------------------
GOOD NEXT STEPS
---------------------------------------------------------------------
• Add depth (L2–L10) features and imbalance slopes
• Probability calibration + cost-aware thresholds
• Replace heuristic fills with a simple matching simulation
• Track feature importances/SHAP
• Batch backfills for longer evaluation windows

License: MIT
