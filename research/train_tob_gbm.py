from pathlib import Path
import os
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------
# CONFIG
# -----------------------
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL   = os.getenv("SYMBOL",   "BTCUSDT")
EVERY    = os.getenv("RESAMPLE_MS", "200ms").strip().lower()
if EVERY.isdigit():
    EVERY = f"{EVERY}ms"

FWD_SECS = int(os.getenv("FWD_SECS", "2"))          # prediction horizon in seconds
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "5.0"))

THRESH_GRID = [0.60, 0.65, 0.70]
EDGE_GRID   = [0.00, 0.05, 0.10]  # micro_edge gate in bps

BASE = Path(__file__).resolve().parents[1]
GLOB = str(
    BASE
    / "data" / "features_tob"
    / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*"
    / f"features_tob_resample-{EVERY}.parquet"
)

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(GLOB).sort("ts")

    # enrich with micro-edge and lags (defensive against missing 'ofi')
    df = df.with_columns([
        pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
        (((pl.col("microprice") - pl.col("mid")) / pl.col("mid")) * 1e4).alias("micro_edge_bps"),
        (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias("d_spread_bps"),
        (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
        (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
        (pl.col("microprice") / pl.col("microprice").shift(1) - 1.0).alias("ret_micro_lag1"),
        # if OFI not present (older files), fall back to zeros so code still runs
        (pl.col("ofi").rolling_sum(window_size=3,  min_samples=1).fill_null(0.0)
         if "ofi" in df.columns else pl.lit(0.0)).alias("ofi_s3"),
        (pl.col("ofi").rolling_sum(window_size=10, min_samples=1).fill_null(0.0)
         if "ofi" in df.columns else pl.lit(0.0)).alias("ofi_s10"),
    ])

    # required target names based on FWD_SECS
    ret_col = f"ret_{FWD_SECS}s"
    dir_col = f"dir_{FWD_SECS}s"

    need = [
        dir_col, ret_col, "spread_bps", "imb",
        "ofi_s3", "ofi_s10",
        "micro_edge_bps", "d_imb", "d_spread_bps",
        "ret_mid_lag1", "ret_mid_lag2", "ret_micro_lag1",
    ]
    # keep only rows where our needed columns are available
    return df.drop_nulls([c for c in need if c in df.columns])


def make_Xy(df: pl.DataFrame):
    feats = [
        "imb", "ofi_s3", "ofi_s10", "micro_edge_bps", "d_imb", "d_spread_bps",
        "spread_bps", "ret_mid_lag1", "ret_mid_lag2", "ret_micro_lag1",
    ]
    feats = [f for f in feats if f in df.columns]

    X = df.select(feats).to_numpy()
    y = df.select(f"dir_{FWD_SECS}s").to_numpy().ravel().astype(int)
    r = df.select(f"ret_{FWD_SECS}s").to_numpy().ravel()
    edge = df.select("micro_edge_bps").to_numpy().ravel()
    return X, y, r, edge, feats


def main():
    df = load_data()
    X, y, ret_fwd, edge, feats = make_Xy(df)
    n = len(y)
    logger.info(
        f"Samples: {n:,} | Features: {feats} | fee_bps={TAKER_FEE_BPS} | "
        f"RESAMPLE={EVERY} | FWD_SECS={FWD_SECS}\n"
    )

    clf = HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=100, random_state=42
    )

    # simple train/test split (last 20% test)
    split = max(1, int(n * 0.8))
    clf.fit(X[:split], y[:split])
    proba = clf.predict_proba(X[split:])
    idx = {c: i for i, c in enumerate(clf.classes_)}
    p_up   = proba[:, idx.get(1, 0)]  if 1  in idx else np.zeros(len(proba))
    p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(proba))

    best = np.maximum(p_up, p_down)
    sign = np.where(p_up >= p_down, 1, -1)

    micro_edge = edge[split:]
    ret_eval   = ret_fwd[split:]
    y_eval     = y[split:]

    for thr in THRESH_GRID:
        for ecut in EDGE_GRID:
            preds = np.where(best >= thr, sign, 0).astype(int)
            preds = np.where(np.abs(micro_edge) >= ecut, preds, 0)

            trade = preds != 0
            hit = (y_eval[trade] == preds[trade]).mean() if trade.any() else np.nan
            gross_bps = 1e4 * ret_eval * preds
            net_bps   = gross_bps - TAKER_FEE_BPS * trade.astype(float)

            print(
                f"[thr={thr:.2f}, edge≥{ecut:.2f}bps] "
                f"hit={hit:.3f} | trade_rate={trade.mean():.3f} | "
                f"avg_gross_bps={gross_bps[trade].mean() if trade.any() else 0.0:.2f} | "
                f"avg_net_bps={net_bps[trade].mean()   if trade.any() else 0.0:.2f}"
            )


if __name__ == "__main__":
    main()
