# Project Closeout Notes (alpha-exec-stack)

## What works ✅
- End-to-end pipeline is runnable locally (ingest → features → train/eval).
- Research scripts run as modules (`python -m research.*`).
- `research.smoke_test` generates synthetic TOB parquet and validates the plumbing.
- Results are persisted under `results/metrics/` (ignored by git).
- CI runs basic checks (smoke test) for reproducibility.

## Known limitations ❗
- Maker “fill” is a heuristic proxy (forward opposite-side notional), not a queue / matching simulation.
- Costs are simplified (fee + spread assumptions), not latency/inventory aware.
- Public feeds only; no deeper book levels / queue position.
- Strategy metrics are research indicators, not “tradable PnL”.

## Next ideas (intentionally postponed) ⏭️
- Add L2–L10 features and slope/pressure signals.
- Probability calibration + cost-aware threshold selection.
- Replace fill proxy with a tiny matching/queue simulator.
- Longer backfills + walk-forward evaluation.
- Save model artifacts + reproducible config snapshots.
