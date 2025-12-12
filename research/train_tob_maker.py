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
if EVERY.isdigit():
    EVERY = f"{EVERY}ms"

FWD_SECS = int(os.getenv("FWD_SECS", "2"))

# Fees (bps)
MAKER_FEE_BPS = float(os.getenv("MAKER_FEE_BPS", "1.0"))
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "5.0"))

# Gates
THRESH_GRID = [0.60, 0.65, 0.70]   # classifier confidence
EDGE_GRID = [0.00, 0.01, 0.02]     # |micro_edge_bps| minimum
FILL_USDT = [25, 50, 100]          # required opposite-side aggressor notional

# CV
K_FOLDS = int(os.getenv("K_FOLDS", "5"))
EMBARGO = 10 if EVERY == "200ms" else 20

BASE = Path(__file__).resolve().parents[1]
GLOB = str(
    BASE
    / "data"
    / "features_tobtrades"
    / f"exchange={EXCHANGE}"
    / f"symbol={SYMBOL}"
    / "date=*"
    / f"features_tobtrades_resample-{EVERY}.parquet"
)

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(GLOB).sort("ts")

    # Core engineered deltas
    df = df.with_columns(
        [
            pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
            ((pl.col("microprice") - pl.col("mid")) / pl.col("mid") * 1e4).alias(
                "micro_edge_bps"
            ),
            (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
            (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias(
                "d_spread_bps"
            ),
            (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
            (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
            (
                pl.col("microprice") / pl.col("microprice").shift(1) - 1.0
            ).alias("ret_micro_lag1"),
        ]
    )

    # OFI: if not present, attempt from size deltas; else zeros
    if "ofi" not in df.columns:
        if "bid_qty" in df.columns and "ask_qty" in df.columns:
            df = df.with_columns(
                (
                    (pl.col("bid_qty") - pl.col("bid_qty").shift(1))
                    - (pl.col("ask_qty") - pl.col("ask_qty").shift(1))
                )
                .fill_null(0.0)
                .alias("ofi")
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("ofi"))

    df = df.with_columns(
        [
            pl.col("ofi").rolling_sum(window_size=3, min_samples=1).alias("ofi_s3"),
            pl.col("ofi").rolling_sum(window_size=10, min_samples=1).alias("ofi_s10"),
        ]
    )

    # Forward notional columns (maker fill proxy): make safe zeros if missing
    for c in [
        "hit_bid_notional_fwd1s",
        "hit_ask_notional_fwd1s",
        "hit_bid_notional_fwd2s",
        "hit_ask_notional_fwd2s",
    ]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))

    return df.drop_nulls()


def make_Xy(df: pl.DataFrame):
    feats = [
        "imb",
        "ofi",
        "micro_edge_bps",
        "d_imb",
        "d_spread_bps",
        "ofi_s3",
        "ofi_s10",
        "spread_bps",
        "ret_mid_lag1",
        "ret_mid_lag2",
        "ret_micro_lag1",
    ]
    feats = [f for f in feats if f in df.columns]

    X = df.select(feats).to_numpy()
    y = df.select(f"dir_{FWD_SECS}s").to_numpy().ravel().astype(int)
    r = df.select(f"ret_{FWD_SECS}s").to_numpy().ravel()
    spr_t = df.select("spread_bps").to_numpy().ravel()

    # future spread at FWD horizon (for exit cost)
    # step = seconds / bucket_ms
    ms = int(EVERY.replace("ms", ""))
    h_fwd = max(1, int(round((FWD_SECS * 1000) / ms)))
    spr_t_fwd = (
        df.select("spread_bps").shift(-h_fwd).to_numpy().ravel()
    )  # future spread at exit

    edge = df.select("micro_edge_bps").to_numpy().ravel()

    # choose fill-gate horizon matching FWD if available, else fall back to fwd1s
    bid_col = f"hit_bid_notional_fwd{FWD_SECS}s"
    ask_col = f"hit_ask_notional_fwd{FWD_SECS}s"
    if bid_col not in df.columns:
        bid_col = "hit_bid_notional_fwd1s"
    if ask_col not in df.columns:
        ask_col = "hit_ask_notional_fwd1s"

    bid_fwd = df.select(bid_col).to_numpy().ravel()
    ask_fwd = df.select(ask_col).to_numpy().ravel()

    return X, y, r, spr_t, spr_t_fwd, edge, bid_fwd, ask_fwd, feats


def purged_kfold(n: int, k: int, emb: int):
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


def maker_entry_taker_exit_netbps(
    ret_fwd: np.ndarray,
    spr_t: np.ndarray,
    spr_t_fwd: np.ndarray,
    preds: np.ndarray,
    fee_maker: float,
    fee_taker: float,
):
    """
    Enter as MAKER at bid/ask (no spread cost at entry),
    exit as TAKER at horizon (pay 0.5*future_spread + taker fee).
    """
    trade = (preds != 0).astype(float)
    gross = 1e4 * ret_fwd * preds
    cost = (0.5 * spr_t_fwd + fee_maker + fee_taker) * trade
    return gross - cost, trade


def main():
    from research.utils_results import save_metrics, save_heatmap  # lazy import

    df = load_data()
    X, y, ret_fwd, spr0, spr_fwd, edge, bid_fwd, ask_fwd, feats = make_Xy(df)
    n = len(y)
    logger.info(
        f"Samples: {n:,} | Features: {feats} | "
        f"maker={MAKER_FEE_BPS}bps taker={TAKER_FEE_BPS}bps | "
        f"RESAMPLE={EVERY} | FWD_SECS={FWD_SECS}\n"
    )

    clf = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        random_state=42,
    )
    folds = purged_kfold(n, K_FOLDS, EMBARGO)

    agg_rows = []

    for thr in THRESH_GRID:
        for ecut in EDGE_GRID:
            for fill_cut in FILL_USDT:
                hitL, trL, gL, nL = [], [], [], []

                for tr, te in folds:
                    if tr.size == 0 or te.size == 0:
                        continue

                    clf.fit(X[tr], y[tr])
                    proba = clf.predict_proba(X[te])

                    idx = {c: i for i, c in enumerate(clf.classes_)}
                    p_up = proba[:, idx.get(1, 0)] if 1 in idx else np.zeros(len(te))
                    p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(te))

                    best = np.maximum(p_up, p_down)
                    sign = np.where(p_up >= p_down, 1, -1)

                    preds = np.where(best >= thr, sign, 0).astype(int)
                    preds = np.where(np.abs(edge[te]) >= ecut, preds, 0)

                    # Fill gate: require opposite-side aggressor notional ≥ threshold
                    long_mask = preds == 1     # we want sellers hitting the bid
                    short_mask = preds == -1   # we want buyers lifting the ask
                    preds[long_mask & (bid_fwd[te] < fill_cut)] = 0
                    preds[short_mask & (ask_fwd[te] < fill_cut)] = 0

                    net, trade = maker_entry_taker_exit_netbps(
                        ret_fwd[te], spr0[te], spr_fwd[te], preds, MAKER_FEE_BPS, TAKER_FEE_BPS
                    )
                    valid = trade == 1
                    gross_bps = 1e4 * ret_fwd[te] * preds

                    hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
                    tr_rate = float(trade.mean())
                    g = float(np.nanmean(gross_bps[valid]) if valid.any() else 0.0)
                    n_ = float(np.nanmean(net[valid]) if valid.any() else 0.0)

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
                        f"[thr={thr:.2f}, edge≥{ecut:.2f}bps, fill≥${fill_cut}] "
                        f"hit={m_hit:.3f} | trade_rate={m_tr:.3f} | "
                        f"avg_gross_bps={m_g:.2f} | avg_net_bps={m_n:.2f}"
                    )

                    agg_rows.append(
                        dict(
                            mode="maker",
                            thr=float(thr),
                            edge=float(ecut),
                            fill_usdt=int(fill_cut),
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
        csv_path = save_metrics(dfm, "maker_grid")
        for fval, chunk in dfm.groupby("fill_usdt"):
            _ = save_heatmap(
                chunk,
                index="thr",
                columns="edge",
                values="avg_net_bps",
                title=f"Maker avg_net_bps (fill≥${fval}, RESAMPLE={EVERY}, FWD={FWD_SECS}s)",
                name=f"maker_netbps_heatmap_fill{fval}",
            )
        logger.info(f"[persist] wrote {csv_path}")


if __name__ == "__main__":
    main()