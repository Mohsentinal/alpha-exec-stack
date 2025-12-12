from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import polars as pl
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------
# CONFIG
# -----------------------
EXCHANGE = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")

EVERY = os.getenv("RESAMPLE_MS", "200ms").strip().lower()
if EVERY.isdigit():  # allow "200" -> "200ms"
    EVERY = f"{EVERY}ms"

FWD_SECS = int(os.getenv("FWD_SECS", "2"))

# taker costs: fee + 0.5 * spread
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "5.0"))

# classifier gates
THRESH_GRID = [0.60, 0.65, 0.70]
EDGE_GRID = [0.00, 0.05, 0.10]  # micro_edge gate in bps

# CV
K_FOLDS = int(os.getenv("K_FOLDS", "5"))
# embargo measured in samples; pick ~1s worth of samples as a guardrail
EMBARGO = 10 if EVERY == "200ms" else 20

BASE = Path(__file__).resolve().parents[1]
GLOB = str(
    BASE
    / "data"
    / "features_tob"
    / f"exchange={EXCHANGE}"
    / f"symbol={SYMBOL}"
    / "date=*"
    / f"features_tob_resample-{EVERY}.parquet"
)

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_data() -> pl.DataFrame:
    """Load time-bucketed TOB features and build a few safe, capped deltas."""
    df = pl.read_parquet(GLOB).sort("ts")

    # enrich with micro-edge and lags (defensive caps)
    df = df.with_columns(
        [
            pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
            (((pl.col("microprice") - pl.col("mid")) / pl.col("mid")) * 1e4).alias(
                "micro_edge_bps"
            ),
            (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
            (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias(
                "d_spread_bps"
            ),
            # ofi rollups if ofi exists; otherwise safe zeros
            (
                pl.col("ofi").rolling_sum(window_size=3, min_samples=1)
                if "ofi" in df.columns
                else pl.lit(0.0)
            ).alias("ofi_s3"),
            (
                pl.col("ofi").rolling_sum(window_size=10, min_samples=1)
                if "ofi" in df.columns
                else pl.lit(0.0)
            ).alias("ofi_s10"),
            (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
            (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
            (
                pl.col("microprice") / pl.col("microprice").shift(1) - 1.0
            ).alias("ret_micro_lag1"),
        ]
    )

    # required target names based on FWD_SECS
    ret_col = f"ret_{FWD_SECS}s"
    dir_col = f"dir_{FWD_SECS}s"
    need = [
        dir_col,
        ret_col,
        "spread_bps",
        "imb",
        "ofi_s3",
        "ofi_s10",
        "micro_edge_bps",
        "d_imb",
        "d_spread_bps",
        "ret_mid_lag1",
        "ret_mid_lag2",
        "ret_micro_lag1",
    ]
    return df.drop_nulls([c for c in need if c in df.columns])


def make_Xy(df: pl.DataFrame):
    feats = [
        "imb",
        "ofi_s3",
        "ofi_s10",
        "micro_edge_bps",
        "d_imb",
        "d_spread_bps",
        "spread_bps",
        "ret_mid_lag1",
        "ret_mid_lag2",
        "ret_micro_lag1",
    ]
    feats = [f for f in feats if f in df.columns]

    X = df.select(feats).to_numpy()
    y = df.select(f"dir_{FWD_SECS}s").to_numpy().ravel().astype(int)
    r = df.select(f"ret_{FWD_SECS}s").to_numpy().ravel()
    edge = df.select("micro_edge_bps").to_numpy().ravel()
    spread = df.select("spread_bps").to_numpy().ravel()
    return X, y, r, edge, spread, feats


def purged_kfold(n: int, k: int, emb: int):
    """Return list of (train_idx, test_idx) tuples with an embargo gap."""
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    starts, s = [], 0
    for fs in sizes:
        starts.append(s)
        s += fs
    ends = [a + b for a, b in zip(starts, sizes)]

    folds = []
    for i in range(k):
        te = np.arange(starts[i], ends[i])
        left, right = max(0, starts[i] - emb), min(n, ends[i] + emb)
        if left == 0 and right == n:
            tr = np.array([], dtype=int)
        else:
            tr = np.concatenate([np.arange(0, left), np.arange(right, n)])
        folds.append((tr, te))
    return folds


def main():
    from research.utils_results import save_metrics, save_heatmap  # lazy import

    df = load_data()
    X, y, ret_fwd, edge, spread, feats = make_Xy(df)
    n = len(y)
    logger.info(
        f"Samples: {n:,} | Features: {feats} | fee_bps={TAKER_FEE_BPS} | "
        f"RESAMPLE={EVERY} | FWD_SECS={FWD_SECS}\n"
    )

    clf = HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100, random_state=42
    )

    folds = purged_kfold(n, K_FOLDS, EMBARGO)

    agg_rows = []

    for thr in THRESH_GRID:
        for ecut in EDGE_GRID:
            hitL, trL, gL, nL = [], [], [], []

            for tr, te in folds:
                if tr.size == 0 or te.size == 0:
                    continue

                clf.fit(X[tr], y[tr])
                proba = clf.predict_proba(X[te])

                # robust class index lookup
                idx = {c: i for i, c in enumerate(clf.classes_)}
                p_up = proba[:, idx.get(1, 0)] if 1 in idx else np.zeros(len(te))
                p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(te))

                best = np.maximum(p_up, p_down)
                sign = np.where(p_up >= p_down, 1, -1)

                preds = np.where(best >= thr, sign, 0).astype(int)
                preds = np.where(np.abs(edge[te]) >= ecut, preds, 0)

                trade = (preds != 0)
                valid = trade

                # pnl in basis points
                gross_bps = 1e4 * ret_fwd[te] * preds
                cost_bps = (TAKER_FEE_BPS + 0.5 * spread[te]) * valid.astype(float)
                net_bps = gross_bps - cost_bps

                hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
                tr_rate = float(valid.mean())
                g = float(np.nanmean(gross_bps[valid]) if valid.any() else 0.0)
                n_ = float(np.nanmean(net_bps[valid]) if valid.any() else 0.0)

                hitL.append(hit)
                trL.append(tr_rate)
                gL.append(g)
                nL.append(n_)

            if hitL:
                m_hit = float(np.nanmean(hitL))
                m_tr = float(np.mean(trL))
                m_g = float(np.mean(gL))
                m_n = float(np.mean(nL))

                print(
                    f"[thr={thr:.2f}, edge≥{ecut:.2f}bps] "
                    f"hit={m_hit:.3f} | trade_rate={m_tr:.3f} | "
                    f"avg_gross_bps={m_g:.2f} | avg_net_bps={m_n:.2f}"
                )

                agg_rows.append(
                    dict(
                        mode="taker",
                        thr=float(thr),
                        edge=float(ecut),
                        hit=m_hit,
                        trade_rate=m_tr,
                        avg_gross_bps=m_g,
                        avg_net_bps=m_n,
                        resample=EVERY,
                        fwd_secs=FWD_SECS,
                    )
                )

    if agg_rows:
        dfm = pd.DataFrame(agg_rows)
        csv_path = save_metrics(dfm, "taker_grid")
        _ = save_heatmap(
            dfm,
            index="thr",
            columns="edge",
            values="avg_net_bps",
            title=f"Taker avg_net_bps (RESAMPLE={EVERY}, FWD={FWD_SECS}s)",
            name="taker_netbps_heatmap",
        )
        logger.info(f"[persist] wrote {csv_path}")


if __name__ == "__main__":
    main()