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

# Parse ms from "200ms"
def _parse_ms(every: str) -> int:
    digits = "".join(ch for ch in every if ch.isdigit())
    return int(digits) if digits else 200

MS = _parse_ms(EVERY)

FWD_SECS      = int(os.getenv("FWD_SECS", "2"))   # prediction & exit horizon for maker eval
H_FWD         = max(1, int(round((FWD_SECS * 1000) / MS)))  # steps forward
K_FOLDS       = 5
EMBARGO       = 10 if MS == 200 else 20          # rough embargo sized in steps

# Fees (bps)
MAKER_FEE_BPS = float(os.getenv("MAKER_FEE_BPS", "1.0"))
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", "5.0"))

# Gates
THRESH_GRID = [0.60, 0.65, 0.70]  # classifier probability
EDGE_GRID   = [0.00, 0.01, 0.02]  # |micro_edge_bps| minimum
FILL_USDT   = [25, 50, 100]       # required same-side aggressive notional within FWD_SECS

BASE = Path(__file__).resolve().parents[1]
GLOB = str(
    BASE
    / "data" / "features_tobtrades"
    / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / "date=*"
    / f"features_tobtrades_resample-{EVERY}.parquet"
)

logger.remove()
logger.add(lambda m: print(m, end=""))


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(GLOB).sort("ts")

    # If 'ofi' is missing, derive a robust proxy from top-of-book sizes
    if "ofi" not in df.columns:
        df = df.with_columns(
            (
                (pl.col("bid_qty") - pl.col("bid_qty").shift(1))
                - (pl.col("ask_qty") - pl.col("ask_qty").shift(1))
            ).fill_null(0.0).alias("ofi")
        )

    # Core engineered features
    df = df.with_columns([
        pl.col("spread_bps").clip(-10.0, 10.0).alias("spread_bps"),
        (((pl.col("microprice") - pl.col("mid")) / pl.col("mid")) * 1e4).alias("micro_edge_bps"),
        (pl.col("imb") - pl.col("imb").shift(1)).alias("d_imb"),
        (pl.col("spread_bps") - pl.col("spread_bps").shift(1)).alias("d_spread_bps"),
        (pl.col("ofi").rolling_sum(window_size=3,  min_samples=1)).alias("ofi_s3"),
        (pl.col("ofi").rolling_sum(window_size=10, min_samples=1)).alias("ofi_s10"),
        (pl.col("mid") / pl.col("mid").shift(1) - 1.0).alias("ret_mid_lag1"),
        (pl.col("mid") / pl.col("mid").shift(2) - 1.0).alias("ret_mid_lag2"),
        (pl.col("microprice") / pl.col("microprice").shift(1) - 1.0).alias("ret_micro_lag1"),
    ])

    # Targets
    ret_col = f"ret_{FWD_SECS}s"
    dir_col = f"dir_{FWD_SECS}s"

    # Forward spread at exit horizon (maker in, taker out)
    df = df.with_columns(
        pl.col("spread_bps").shift(-H_FWD).alias("spread_bps_fwd")
    )

    # Ensure forward hit notional columns matching horizon exist (else safe zeros)
    bid_fwd_col = f"hit_bid_notional_fwd{FWD_SECS}s"
    ask_fwd_col = f"hit_ask_notional_fwd{FWD_SECS}s"
    if bid_fwd_col not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias(bid_fwd_col))
    if ask_fwd_col not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias(ask_fwd_col))

    need = [
        dir_col, ret_col,
        "spread_bps", "spread_bps_fwd", "imb", "ofi",
        "micro_edge_bps", "d_imb", "d_spread_bps",
        "ofi_s3", "ofi_s10", "ret_mid_lag1", "ret_mid_lag2", "ret_micro_lag1",
        bid_fwd_col, ask_fwd_col
    ]
    return df.drop_nulls([c for c in need if c in df.columns])


def make_Xy(df: pl.DataFrame):
    feats = [
        "imb", "ofi", "micro_edge_bps", "d_imb", "d_spread_bps",
        "ofi_s3", "ofi_s10", "spread_bps",
        "ret_mid_lag1", "ret_mid_lag2", "ret_micro_lag1",
    ]
    feats = [f for f in feats if f in df.columns]

    X = df.select(feats).to_numpy()
    y = df.select(f"dir_{FWD_SECS}s").to_numpy().ravel().astype(int)
    r = df.select(f"ret_{FWD_SECS}s").to_numpy().ravel()
    spr_now  = df.select("spread_bps").to_numpy().ravel()
    spr_fwd  = df.select("spread_bps_fwd").to_numpy().ravel()
    edge     = df.select("micro_edge_bps").to_numpy().ravel()
    bid_fwd  = df.select(f"hit_bid_notional_fwd{FWD_SECS}s").to_numpy().ravel()
    ask_fwd  = df.select(f"hit_ask_notional_fwd{FWD_SECS}s").to_numpy().ravel()
    return X, y, r, spr_now, spr_fwd, edge, bid_fwd, ask_fwd, feats


def purged_kfold(n, k, emb):
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
        tr = (
            np.concatenate([np.arange(0, left), np.arange(right, n)])
            if (left > 0 or right < n) else np.array([], int)
        )
        folds.append((tr, te))
    return folds


def maker_entry_taker_exit_netbps(ret_fwd, spr_t, spr_t_fwd, preds, fee_maker, fee_taker):
    """
    Enter as MAKER at bid/ask (no spread cost at entry),
    exit as TAKER at FWD horizon (pay 0.5 * spread at exit).
    """
    trade = (preds != 0).astype(float)
    gross = 1e4 * ret_fwd * preds
    cost  = (0.5 * spr_t_fwd + fee_maker + fee_taker) * trade
    return gross - cost, trade


def main():
    df = load_data()
    X, y, ret_fwd, spr0, sprF, edge, bid_fwd, ask_fwd, feats = make_Xy(df)
    n = len(y)
    logger.info(
        f"Samples: {n:,} | Features: {feats} | maker={MAKER_FEE_BPS}bps taker={TAKER_FEE_BPS}bps | "
        f"RESAMPLE={EVERY} | FWD_SECS={FWD_SECS}\n"
    )

    clf = HistGradientBoostingClassifier(
        max_depth=7, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=100, random_state=42,
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
                    p_up   = proba[:, idx.get(1, 0)]  if 1  in idx else np.zeros(len(te))
                    p_down = proba[:, idx.get(-1, 0)] if -1 in idx else np.zeros(len(te))

                    best = np.maximum(p_up, p_down)
                    sign = np.where(p_up >= p_down, 1, -1)
                    preds = np.where(best >= thr, sign, 0).astype(int)

                    # micro-edge gate
                    preds = np.where(np.abs(edge[te]) >= ecut, preds, 0)

                    # FILL GATE at the same horizon FWD_SECS
                    long_mask  = preds == 1
                    short_mask = preds == -1
                    preds[ long_mask  & (bid_fwd[te] <  fill_cut) ] = 0   # need sellers hitting bid
                    preds[ short_mask & (ask_fwd[te] <  fill_cut) ] = 0   # need buyers hitting ask

                    net, trade = maker_entry_taker_exit_netbps(
                        ret_fwd[te], spr0[te], sprF[te], preds, MAKER_FEE_BPS, TAKER_FEE_BPS
                    )
                    valid = trade == 1
                    hit = (y[te][valid] == preds[valid]).mean() if valid.any() else np.nan
                    gross_bps = 1e4 * ret_fwd[te] * preds

                    hitL.append(hit)
                    trL.append(float(trade.mean()))
                    gL.append(float(np.nanmean(gross_bps[valid]) if valid.any() else 0.0))
                    nL.append(float(np.nanmean(net[valid])         if valid.any() else 0.0))

                if hitL:
                    print(
                        f"[thr={thr:.2f}, edge≥{ecut:.2f}bps, fill≥${fill_cut}, fwd={FWD_SECS}s] "
                        f"hit={np.nanmean(hitL):.3f} | trade_rate={np.mean(trL):.3f} | "
                        f"avg_gross_bps={np.mean(gL):.2f} | avg_net_bps={np.mean(nL):.2f}"
                    )


if __name__ == "__main__":
    main()
