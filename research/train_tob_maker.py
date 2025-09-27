from pathlib import Path
import os
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

# =========================
# CONFIG (env-overridable)
# =========================
EXCHANGE = "binance"
SYMBOL   = "BTCUSDT"

# Resample step used by features, e.g. 200ms or 100ms
RESAMPLE_MS = int(os.getenv("RESAMPLE_MS", "200"))
EVERY       = f"{RESAMPLE_MS}ms"

# Forward window (in seconds) to check for aggressive fills that would hit our resting order
FWD_SECS = int(os.getenv("FWD_SECS", "2"))
H_1S = max(1, 1000 // RESAMPLE_MS)          # steps per 1 second
H_FWD = max(1, (1000 * FWD_SECS) // RESAMPLE_MS)

# CV settings
K_FOLDS  = 5
EMBARGO  = 2 * H_1S                          # keep ~2s embargo by default

# Fees (bps)
MAKER_FEE_BPS = 1.0
TAKER_FEE_BPS = 5.0

# Gates / grids
THRESH_GRID = [0.60, 0.65, 0.70]             # classifier probability
EDGE_GRID   = [0.00, 0.01, 0.02]             # |micro_edge_bps| minimum
FILL_USDT   = [25, 50, 100]                  # required same-side aggressive notional within FWD_SECS

# Paths
BASE = Path(__file__).resolve().parents[1]
GLOB = str(
    BASE / "data" / "features_tobtrades" /
    f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*" /
    f"features_tobtrades_resample-{EVERY}.parquet"
)

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(GLOB).sort("ts")

    # Simple cleaning / feature engineering
    df = df.with_columns([
        pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
        pl.col("ret_1s").clip(-0.01, 0.01).alias("ret_1s"),  # cap for robustness
        (((pl.col("microprice") - pl.col("mid")) / pl.col("mid")) * 1e4).alias("micro_edge_bps"),
        (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias("d_spread_bps"),
        pl.col("ofi").rolling_sum(window_size=3,  min_samples=1).alias("ofi_s3"),
        pl.col("ofi").rolling_sum(window_size=10, min_samples=1).alias("ofi_s10"),
        (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
        (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
        (pl.col("microprice") / pl.col("microprice").shift(1) - 1.0).alias("ret_micro_lag1"),
    ])

    # forward-fill column names based on FWD_SECS
    hit_bid_col = f"hit_bid_notional_fwd{FWD_SECS}s"
    hit_ask_col = f"hit_ask_notional_fwd{FWD_SECS}s"

    need = [
        "dir_1s", "ret_1s", "spread_bps", "imb", "ofi",
        "micro_edge_bps", "d_imb", "d_spread_bps",
        "ofi_s3", "ofi_s10", "ret_mid_lag1", "ret_mid_lag2", "ret_micro_lag1",
        hit_bid_col, hit_ask_col
    ]
    return df.drop_nulls(need)


def make_Xy(df: pl.DataFrame):
    feats = [
        "imb", "ofi", "micro_edge_bps", "d_imb", "d_spread_bps",
        "ofi_s3", "ofi_s10", "spread_bps", "ret_mid_lag1",
        "ret_mid_lag2", "ret_micro_lag1"
    ]
    X = df.select(feats).to_numpy()
    y = df.select("dir_1s").to_numpy().ravel().astype(int)
    r = df.select("ret_1s").to_numpy().ravel()

    spr_t  = df.select("spread_bps").to_numpy().ravel()
    # spread 1s later (used as part of taker exit cost)
    spr_t1 = df.select("spread_bps").shift(-H_1S).to_numpy().ravel()

    edge = df.select("micro_edge_bps").to_numpy().ravel()

    hit_bid_col = f"hit_bid_notional_fwd{FWD_SECS}s"
    hit_ask_col = f"hit_ask_notional_fwd{FWD_SECS}s"
    bid_fwd = df.select(hit_bid_col).to_numpy().ravel()
    ask_fwd = df.select(hit_ask_col).to_numpy().ravel()

    return X, y, r, spr_t, spr_t1, edge, bid_fwd, ask_fwd, feats


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
        tr = np.concatenate([np.arange(0, left), np.arange(right, n)]) if (left > 0 or right < n) else np.array([], int)
        folds.append((tr, te))
    return folds


def maker_entry_taker_exit_netbps(ret_1s, spr_t, spr_t1, preds, fee_maker, fee_taker):
    """
    Enter as MAKER at bid/ask (no spread paid on entry).
    Exit as TAKER after 1s (proxy), so pay 0.5*spread at exit + fees.
    """
    trade = (preds != 0).astype(float)
    gross = 1e4 * ret_1s * preds
    cost  = (0.5 * spr_t1 + fee_maker + fee_taker) * trade
    return gross - cost, trade


def main():
    df = load_data()
    X, y, ret1, spr0, spr1, edge, bid_fwd, ask_fwd, feats = make_Xy(df)
    n = len(y)

    logger.info(
        f"Samples: {n:,} | Features: {feats} | "
        f"maker={MAKER_FEE_BPS}bps taker={TAKER_FEE_BPS}bps | "
        f"RESAMPLE={EVERY} | FWD_SECS={FWD_SECS}\n"
    )

    clf = HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=100, random_state=42
    )
    folds = purged_kfold(n, K_FOLDS, EMBARGO)

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
                    p_up   = proba[:, idx.get(1, 0)] if 1 in idx else np.zeros(len(te))
                    p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(te))

                    best = np.maximum(p_up, p_down)
                    sign = np.where(p_up >= p_down, 1, -1)
                    preds = np.where(best >= thr, sign, 0).astype(int)

                    # micro-edge gate
                    preds = np.where(np.abs(edge[te]) >= ecut, preds, 0)

                    # FILL gate: require future opposite-side aggressive notional ≥ threshold
                    long_mask  = preds == 1
                    short_mask = preds == -1
                    preds[ long_mask  & (bid_fwd[te] <  fill_cut) ] = 0   # need sellers hitting the BID
                    preds[ short_mask & (ask_fwd[te] <  fill_cut) ] = 0   # need buyers hitting the ASK

                    net, trade = maker_entry_taker_exit_netbps(
                        ret1[te], spr0[te], spr1[te], preds, MAKER_FEE_BPS, TAKER_FEE_BPS
                    )
                    valid = trade == 1
                    hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
                    gross_bps = 1e4 * ret1[te] * preds

                    hitL.append(hit)
                    trL.append(float(trade.mean()))
                    gL.append(float(np.nanmean(gross_bps[valid]) if valid.any() else 0.0))
                    nL.append(float(np.nanmean(net[valid]) if valid.any() else 0.0))

                if hitL:
                    print(
                        f"[thr={thr:.2f}, edge≥{ecut:.2f}bps, fill≥${fill_cut}, "
                        f"fwd={FWD_SECS}s] "
                        f"hit={np.nanmean(hitL):.3f} | trade_rate={np.mean(trL):.3f} | "
                        f"avg_gross_bps={np.mean(gL):.2f} | avg_net_bps={np.mean(nL):.2f}"
                    )


if __name__ == "__main__":
    main()
